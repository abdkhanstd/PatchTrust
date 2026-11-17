"""Evaluate a base or PatchTrust checkpoint on CamVid test under clean, seen and
held-out corruptions (3 severities each, deterministic seeds) and on KITTI Road
(zero-shot binary road). Saves raw per-image counts for paired statistics."""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CamVid, FixedCorrupted, KittiRoad, NUM_CLASSES, IGNORE, make_dataset
from .corruptions import SEEN, HELDOUT
from .models import Base, PatchTrust, normalize
from .metrics import per_image_iu, miou_from, boundary_counts, bf1_from, CalibAcc, binary_auroc_from_hist

ROAD = 3


def load_model(ckpt, base_ckpt=None, gate_override=None, device='cuda', gate_thresh=None):
    ck = torch.load(ckpt, map_location='cpu')
    a = ck['args']
    if a['mode'] == 'heads':
        bck = torch.load(base_ckpt or a['base'], map_location='cpu')
        base = Base(NUM_CLASSES, pretrained_backbone=False, arch=bck['args'].get('arch', 'large'))
        base.load_state_dict(bck['model'])
        m = PatchTrust(base, gate=gate_override or a['gate'], use_cues=bool(a.get('cues', 1)), gate_thresh=gate_thresh,
                       corr_pb=bool(a.get('corr_pb', 1)), rel_feats=bool(a.get('rel_feats', 1)))
        m.heads.load_state_dict(ck['model'])
    else:
        base = Base(NUM_CLASSES, pretrained_backbone=False, arch=a.get('arch', 'large'))
        base.load_state_dict(ck['model'])
        m = PatchTrust(base, gate='none')
    return m.to(device).eval()


@torch.no_grad()
def fit_temperature(model, ds, device):
    logits, labs = [], []
    for x, y in DataLoader(ds, batch_size=16, num_workers=4):
        out = model(normalize(x.to(device)))
        lo = out['final'][:, :, ::4, ::4]
        yy = y.to(device)[:, ::4, ::4]
        v = yy != IGNORE
        logits.append(lo.permute(0, 2, 3, 1)[v].float())
        labs.append(yy[v])
    L, Y = torch.cat(logits), torch.cat(labs)
    best = (1e9, 1.0)
    for T in np.arange(0.5, 3.01, 0.05):
        nll = F.cross_entropy(L / T, Y).item()
        best = min(best, (nll, float(T)))
    return best[1]


@torch.no_grad()
def run_condition(model, ds, device, T, gate_oracle=False, ref=None):
    I, U, C, V = [], [], [], []
    bnd = np.zeros(4)
    cal = {k: CalibAcc() for k in ['msp', 'msp_ts', 'q', 'base_msp']}
    relacc = CalibAcc()  # r as predictor of base correctness
    flips = np.zeros(4)  # base-correct, base-correct->wrong, base-wrong, base-wrong->correct
    gstat = np.zeros(4)  # sum g on base-correct, n, sum g on base-wrong, n
    rflips = np.zeros(4)  # same as flips but relative to a reference model (e.g. the clean A0 model)
    for x, y in DataLoader(ds, batch_size=16, num_workers=4):
        x, y = normalize(x.to(device)), y.to(device)
        valid = y != IGNORE
        if gate_oracle:
            with torch.no_grad():
                bl = model.base(x)
            out = model(x, oracle_correct=(bl.argmax(1) == y))
        else:
            out = model(x)
        lf, lb = out['final'].float(), out['base'].float()
        pred, pb = lf.argmax(1), lb.argmax(1)
        i, u, c, v = per_image_iu(pred, y, NUM_CLASSES)
        I.append(i.cpu()); U.append(u.cpu()); C.append(c.cpu()); V.append(v.cpu())
        bnd += boundary_counts(pred, y)
        prob = lf.softmax(1)
        correct = pred == y
        cal['msp'].add(prob.max(1)[0], correct, valid, prob, y)
        pts = (lf / T).softmax(1)
        cal['msp_ts'].add(pts.max(1)[0], correct, valid, pts, y)
        pbp = lb.softmax(1)
        cal['base_msp'].add(pbp.max(1)[0], pb == y, valid)
        if 'q' in out:
            cal['q'].add(out['q'].squeeze(1), correct, valid)
            relacc.add(out['r'].squeeze(1), pb == y, valid)
        bc, bw = (pb == y) & valid, (pb != y) & valid
        flips += [bc.sum().item(), (bc & ~correct).sum().item(), bw.sum().item(), (bw & correct).sum().item()]
        if ref is not None:
            pr = ref(x)['final'].argmax(1)
            rc, rw = (pr == y) & valid, (pr != y) & valid
            rflips += [rc.sum().item(), (rc & ~correct).sum().item(), rw.sum().item(), (rw & correct).sum().item()]
        if 'g' in out:
            g = out['g'].squeeze(1)
            gstat += [g[bc].sum().item(), bc.sum().item(), g[bw].sum().item(), bw.sum().item()]
    I, U = torch.cat(I).numpy(), torch.cat(U).numpy()
    miou, iou = miou_from(I.sum(0), U.sum(0))
    res = {'miou': miou, 'iou': iou.tolist(), 'pixacc': float(torch.cat(C).sum() / torch.cat(V).sum()),
           'bf1': bf1_from(bnd), 'T': T, 'flips': flips.tolist(), 'gstat': gstat.tolist(), 'ref_flips': rflips.tolist()}
    for k, a in cal.items():
        if a.n == 0:
            continue
        res[k] = {'ece': a.ece(), 'aurc': a.aurc(), 'auroc': binary_auroc_from_hist(a.rc_n, a.rc_c)}
        if k in ('msp', 'msp_ts'):
            res[k].update(brier=a.brier(), nll=a.nll())
    if relacc.n:
        res['r_vs_base'] = {'ece': relacc.ece(), 'auroc': binary_auroc_from_hist(relacc.rc_n, relacc.rc_c),
                            'brier_bin': None}
    raw = {'I': I, 'U': U, 'C': torch.cat(C).numpy(), 'V': torch.cat(V).numpy(),
           'rc_msp_n': cal['msp'].rc_n, 'rc_msp_c': cal['msp'].rc_c,
           'rc_q_n': cal['q'].rc_n, 'rc_q_c': cal['q'].rc_c}
    return res, raw


@torch.no_grad()
def run_kitti(model, ds, device, cond):
    tot = np.zeros(3)  # I, U, (unused)
    per = []
    for x, y in DataLoader(ds, batch_size=8, num_workers=4):
        x, y = normalize(x.to(device)), y.to(device)
        out = model(x)
        pr = (out['final'].argmax(1) == ROAD).long()
        v = y != IGNORE
        i = ((pr == 1) & (y == 1) & v).flatten(1).sum(1)
        u = (((pr == 1) | (y == 1)) & v).flatten(1).sum(1)
        per.append(torch.stack([i, u], 1).cpu())
    per = torch.cat(per).numpy()
    return {'road_iou': float(per[:, 0].sum() / per[:, 1].sum())}, per


class KittiCorr(torch.utils.data.Dataset):
    def __init__(self, base, ctype, sev):
        self.b, self.c, self.s = base, ctype, sev

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        from .corruptions import ALL, apply_corruption
        x, y = self.b[i]
        if self.c is None:
            return x, y
        rng = np.random.default_rng([2026, ALL.index(self.c), self.s, i])
        return torch.from_numpy(apply_corruption(x.numpy(), self.c, self.s, rng)), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default=None)
    ap.add_argument('--gate_override', default=None)
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--kitti', default=os.path.expanduser('~/patchtrust/data/data_road'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--severities', default='1,2,3')
    ap.add_argument('--gate_thresh', type=float, default=None)
    ap.add_argument('--dataset', default='camvid', choices=['camvid', 'kitti'])
    ap.add_argument('--cache', default=os.path.expanduser('~/patchtrust/data/cache'))
    ap.add_argument('--ref', default=None, help='reference base checkpoint for negative-flip analysis')
    args = ap.parse_args()
    device = torch.device('cuda')
    model = load_model(args.ckpt, args.base, args.gate_override, device, args.gate_thresh)
    oracle = args.gate_override == 'oracle'
    ref = load_model(args.ref, device=device) if args.ref else None
    T = fit_temperature(model, FixedCorrupted(make_dataset(args.dataset, args.data, 'val')), device) if not oracle else 1.0
    te = make_dataset(args.dataset, args.data, args.split)
    conds = [('clean', None, 0)]
    for c in SEEN + HELDOUT:
        for s in [int(v) for v in args.severities.split(',')]:
            conds.append((f'{c}@{s}', c, s))
    results, raws = {}, {}
    for name, c, s in conds:
        r, raw = run_condition(model, FixedCorrupted(te, c, s, cache=args.cache), device, T, oracle, ref)
        results[name] = r
        for k, v in raw.items():
            raws[f'{name}|{k}'] = v
        print(name, f"mIoU={r['miou']:.4f} bf1={r['bf1']:.4f} ece={r['msp']['ece']:.4f}", flush=True)
    if not oracle and args.dataset == 'camvid' and os.path.isdir(args.kitti):
        kd = KittiRoad(args.kitti)
        results['kitti'] = {}
        for c in [None] + SEEN + HELDOUT:
            r, per = run_kitti(model, KittiCorr(kd, c, 2), device, c)
            results['kitti'][c or 'clean'] = r
            raws[f'kitti|{c or "clean"}'] = per
        print('kitti clean road IoU', results['kitti']['clean'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(results, open(args.out + '.json', 'w'), indent=1, default=float)
    np.savez_compressed(args.out + '.npz', **raws)


if __name__ == '__main__':
    main()
