"""Training for base segmenters (A0/A1/A1c) and PatchTrust heads (A2-A6).

Base modes:  clean  - CE on clean crops                        (A0)
             aug    - CE on crops corrupted with p=0.5          (A1)
             augjsd - CE(clean)+CE(corrupt)+12*JSD (AugMix-style consistency) (A1c)
Head modes:  --base ckpt --gate {ungated,entropy,learned,learned_unsup} [--cons 0/1]
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CamVid, FixedCorrupted, NUM_CLASSES, IGNORE, make_dataset
from .corruptions import SEEN
from .models import Base, PatchTrust, normalize, count_params
from .metrics import per_image_iu, miou_from


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)


@torch.no_grad()
def evaluate_val(model, val_sets, device, is_pt):
    model.eval()
    res = {}
    for name, ds in val_sets.items():
        I = np.zeros(NUM_CLASSES)
        U = np.zeros(NUM_CLASSES)
        for x, y in DataLoader(ds, batch_size=16, num_workers=4):
            x, y = normalize(x.to(device)), y.to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(x)
            lo = out['final'] if is_pt else out
            i, u, _, _ = per_image_iu(lo.argmax(1), y, NUM_CLASSES)
            I += i.sum(0).cpu().numpy()
            U += u.sum(0).cpu().numpy()
        res[name] = miou_from(I, U)[0]
    clean = res['clean']
    corr = float(np.mean([v for k, v in res.items() if k != 'clean']))
    return clean, corr, res


def kl_masked(p_t, logit_s, mask):
    """KL(p_t || softmax(logit_s)) averaged over masked pixels."""
    logq = F.log_softmax(logit_s.float(), 1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-8)) - logq)).sum(1)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def jsd(l1, l2):
    p1, p2 = F.softmax(l1.float(), 1), F.softmax(l2.float(), 1)
    m = (0.5 * (p1 + p2)).clamp(1e-7, 1).log()
    return 0.5 * (F.kl_div(m, p1, reduction='none').sum(1) + F.kl_div(m, p2, reduction='none').sum(1)).mean()


def pt_losses(model, x, y, args, teacher_p=None):
    out = model(x, y=y)
    valid = y != IGNORE
    seg = F.cross_entropy(out['final'], y, ignore_index=IGNORE)
    loss = seg
    logs = {'seg': seg.item()}
    if model.gate in ('learned', 'learned_tf', 'learned_unsup', 'ungated', 'entropy', 'select'):
        cb = (out['base'].argmax(1) == y).float()
        cf = (out['final'].detach().argmax(1) == y).float()
        bce = lambda lg, t: F.binary_cross_entropy_with_logits(lg.squeeze(1)[valid], t[valid])
        lr = bce(out['r_logit'], cb) + bce(out['q_logit'], cf)
        loss = loss + args.lam1 * lr
        logs['rel'] = lr.item()
    if model.gate == 'select' and 'final_sel' in out:
        # selector target: 1 where the correction fixes a base error, 0 where it breaks a correct base pixel
        pb, pc = out['base'].argmax(1), (out['base'] + out['dl'].detach()).argmax(1)
        helps, harms = (pb != y) & (pc == y) & valid, (pb == y) & (pc != y) & valid
        m = helps | harms
        if m.any():
            ls = F.binary_cross_entropy_with_logits(out['s_logit'].squeeze(1)[m], helps[m].float())
        else:
            ls = out['s_logit'].sum() * 0
        lsel = F.cross_entropy(out['final_sel'], y, ignore_index=IGNORE)
        loss = loss + ls + lsel
        logs['sel'] = ls.item()
    return out, loss, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='clean', choices=['clean', 'aug', 'augjsd', 'heads'])
    ap.add_argument('--base', default=None, help='base checkpoint for heads mode')
    ap.add_argument('--gate', default='learned')
    ap.add_argument('--gate_grad', type=int, default=0)
    ap.add_argument('--cues', type=int, default=1)
    ap.add_argument('--cons', type=int, default=1)
    ap.add_argument('--lam1', type=float, default=0.2)
    ap.add_argument('--lam2', type=float, default=0.1)
    ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--clean_loss', type=int, default=1, help='heads: also train on the clean view')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--dataset', default='camvid', choices=['camvid', 'kitti'])
    ap.add_argument('--arch', default='large', choices=['large', 'small'])
    ap.add_argument('--corr_pb', type=int, default=1, help='corrector receives the base softmax')
    ap.add_argument('--rel_feats', type=int, default=1, help='reliability head receives encoder features')
    ap.add_argument('--train_part', default='all', choices=['all', 'decoder'], help='base modes: train all or only the LR-ASPP decoder')
    ap.add_argument('--init', default=None, help='base modes: initialise from this base checkpoint (fine-tune continuation)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heads = args.mode == 'heads'
    corrupt_p = {'clean': 0.0, 'aug': 0.5, 'augjsd': 1.0, 'heads': 1.0}[args.mode]
    tr = make_dataset(args.dataset, args.data, 'train', train=True, corrupt_p=corrupt_p, subset=args.subset, seed=args.seed)
    va = make_dataset(args.dataset, args.data, 'val')
    val_sets = {'clean': FixedCorrupted(va)}
    for c in SEEN:
        val_sets[c] = FixedCorrupted(va, c, 2, seed=7)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                    worker_init_fn=worker_init, persistent_workers=True)

    if heads:
        bck = torch.load(args.base, map_location='cpu')
        base = Base(NUM_CLASSES, pretrained_backbone=False, arch=bck['args'].get('arch', 'large'))
        base.load_state_dict(bck['model'])
        for p in base.parameters():
            p.requires_grad_(False)
        model = PatchTrust(base, gate=args.gate, gate_grad=bool(args.gate_grad), use_cues=bool(args.cues),
                           corr_pb=bool(args.corr_pb), rel_feats=bool(args.rel_feats)).to(device)
        params = [p for p in model.heads.parameters()]
    else:
        model = Base(NUM_CLASSES, pretrained_backbone=True, arch=args.arch).to(device)
        if args.init:
            model.load_state_dict(torch.load(args.init, map_location='cpu')['model'])
        params = list(model.parameters())
        if args.train_part == 'decoder':  # parameter-matched control: only the LR-ASPP decoder is trained
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            params = list(model.classifier.parameters())
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(it, total) / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    start_ep, best, bad = 0, -1.0, 0
    hist = []
    last = os.path.join(args.out, 'last.pt')
    if os.path.exists(last):  # checkpoint recovery
        ck = torch.load(last, map_location='cpu')
        (model.heads if heads else model).load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start_ep, best, bad, hist = ck['epoch'] + 1, ck['best'], ck['bad'], ck['hist']
        print('resumed from epoch', start_ep)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        if not heads and args.train_part == 'decoder':
            model.backbone.eval()
        agg = {}
        for xc, xx, y, cid in dl:
            xc, xx, y = normalize(xc.to(device, non_blocking=True)), normalize(xx.to(device, non_blocking=True)), y.to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                if not heads:
                    if args.mode == 'augjsd':
                        lc, lx = model(xc), model(xx)
                        loss = F.cross_entropy(lc.float(), y, ignore_index=IGNORE) + F.cross_entropy(lx.float(), y, ignore_index=IGNORE) + 12.0 * jsd(lc, lx)
                    else:
                        loss = F.cross_entropy(model(xx).float(), y, ignore_index=IGNORE)
                    logs = {'loss': loss.item()}
                else:
                    out_x, loss, logs = pt_losses(model, xx, y, args)
                    if args.clean_loss:
                        out_c, lc, _ = pt_losses(model, xc, y, args)
                        loss = loss + lc
                    if args.cons:
                        if not args.clean_loss:
                            with torch.no_grad():
                                out_c = {'base': F.interpolate(model.base.forward_all(xc)[2], size=xc.shape[-2:], mode='bilinear', align_corners=False).float()}
                        p_t = out_c['base'].detach().softmax(1)
                        conf, am = p_t.max(1)
                        mask = (conf > args.tau) & (am == y) & (y != IGNORE)
                        lcons = kl_masked(p_t, out_x['final'], mask)
                        loss = loss + args.lam2 * lcons
                        logs['cons'] = lcons.item()
                    logs['loss'] = loss.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / len(dl)
        rec = {'epoch': ep, 'time': time.time() - t0, **agg}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            clean, corr, res = evaluate_val(model, val_sets, device, heads)
            score = 0.5 * (clean + corr)
            rec.update(val_clean=clean, val_corr=corr, val_score=score)
            if score > best:
                best, bad = score, 0
                torch.save({'model': (model.heads if heads else model).state_dict(), 'args': vars(args), 'epoch': ep,
                            'val': res}, os.path.join(args.out, 'best.pt'))
            else:
                bad += args.eval_every
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({'model': (model.heads if heads else model).state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'scaler': scaler.state_dict(), 'epoch': ep, 'best': best,
                    'bad': bad, 'hist': hist}, last)
        if bad >= args.patience:
            print('early stop at epoch', ep)
            break
    summary = {'args': vars(args), 'best_val_score': best, 'train_time_s': time.time() - t0,
               'peak_mem_MB': torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else None,
               'trainable_params': n_train, 'epochs_run': len(hist), 'hist': hist}
    json.dump(summary, open(os.path.join(args.out, 'train_summary.json'), 'w'), indent=1)
    os.remove(last)


if __name__ == '__main__':
    main()
