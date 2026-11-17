"""Mechanism analysis of a trained PatchTrust model on the cached test conditions.

* Self-gating: rate at which the ungated correction (L_b + dL) changes the base
  label, and mean |dL| (max over classes), per bin of base max-probability.
* Detection vs correction decomposition:
    fix rate   = share of base-wrong pixels that L_b + dL gets right
                 (the corrector applied exactly where it is needed, oracle gate)
    harm rate  = share of base-correct pixels that L_b + dL gets wrong
    detector   = r < 0.5 flags a predicted base error: precision and recall
    learned detector + perfect corrector: the true label on pixels with r < 0.5
* Spatial vs image-level detection: mean over images of the within-image AUROC of
  r and of base max-prob for separating correct from wrong base pixels, and the
  Spearman correlation across images between mean (1 - score) and the base error rate.
"""
import argparse
import json
import os
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .data import CamVid, FixedCorrupted, NUM_CLASSES, IGNORE
from .corruptions import SEEN, HELDOUT
from .models import normalize
from .evaluate import load_model
from .coverage import miou_conf

BINS = np.linspace(0, 1, 11)


def auroc(score, correct):
    """AUROC of score for correct (positive) vs wrong pixels."""
    if correct.all() or (~correct).all():
        return np.nan
    r = stats.rankdata(score)
    n1 = correct.sum()
    n0 = len(correct) - n1
    return (r[correct].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


@torch.no_grad()
def analyse(model, ds, device):
    nb = len(BINS) - 1
    ch_n, ch_c, ch_m = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    cnt = dict(bw=0, bc=0, fix=0, harm=0, fix_pt=0, harm_pt=0, det=0, det_tp=0)
    Y, PU, PB = [], [], []
    au_r, au_m, img_err, img_r, img_m = [], [], [], [], []
    for x, y in DataLoader(ds, batch_size=16, num_workers=4):
        x, y = normalize(x.to(device)), y.to(device)
        out = model(x)
        lb, dl, r = out['base'], out['dl'], out['r'].squeeze(1)
        pb = lb.argmax(1)
        pu = (lb + dl).argmax(1)  # ungated correction
        pf = out['final'].argmax(1)
        msp = lb.softmax(1).max(1)[0]
        v = y != IGNORE
        bc, bw = (pb == y) & v, (pb != y) & v
        cnt['bw'] += bw.sum().item(); cnt['bc'] += bc.sum().item()
        cnt['fix'] += (bw & (pu == y)).sum().item(); cnt['harm'] += (bc & (pu != y)).sum().item()
        cnt['fix_pt'] += (bw & (pf == y)).sum().item(); cnt['harm_pt'] += (bc & (pf != y)).sum().item()
        det = (r < 0.5) & v
        cnt['det'] += det.sum().item(); cnt['det_tp'] += (det & bw).sum().item()
        b = torch.bucketize(msp[v], torch.tensor(BINS[1:-1], device=device))
        chg = (pu != pb)[v].float()
        mag = dl.abs().max(1)[0][v]
        ch_n += torch.bincount(b, minlength=nb).cpu().numpy()
        ch_c += torch.bincount(b, weights=chg, minlength=nb).cpu().numpy()
        ch_m += torch.bincount(b, weights=mag, minlength=nb).cpu().numpy()
        po = torch.where(det, y, pb)  # learned detector + perfect corrector
        Y.append(y[v]); PU.append(po[v]); PB.append(pb[v])
        for i in range(x.shape[0]):
            vi = v[i]
            c = bc[i][vi].cpu().numpy()
            au_r.append(auroc(r[i][vi].float().cpu().numpy(), c))
            au_m.append(auroc(msp[i][vi].float().cpu().numpy(), c))
            img_err.append(1 - c.mean()); img_r.append(1 - r[i][vi].mean().item()); img_m.append(1 - msp[i][vi].mean().item())
    Y, PU, PB = torch.cat(Y), torch.cat(PU), torch.cat(PB)
    return {
        'change_rate_by_msp': (ch_c / np.maximum(ch_n, 1)).tolist(), 'dl_mag_by_msp': (ch_m / np.maximum(ch_n, 1)).tolist(),
        'pix_by_msp': ch_n.tolist(),
        'fix_rate': cnt['fix'] / max(cnt['bw'], 1), 'harm_rate': cnt['harm'] / max(cnt['bc'], 1),
        'fix_rate_pt': cnt['fix_pt'] / max(cnt['bw'], 1), 'harm_rate_pt': cnt['harm_pt'] / max(cnt['bc'], 1),
        'det_precision': cnt['det_tp'] / max(cnt['det'], 1), 'det_recall': cnt['det_tp'] / max(cnt['bw'], 1),
        'miou_base': miou_conf(Y, PB), 'miou_det_perfectcorr': miou_conf(Y, PU),
        'within_auroc_r': float(np.nanmean(au_r)), 'within_auroc_msp': float(np.nanmean(au_m)),
        'image_spearman_r': float(stats.spearmanr(img_r, img_err)[0]), 'image_spearman_msp': float(stats.spearmanr(img_m, img_err)[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--cache', default=os.path.expanduser('~/patchtrust/data/cache'))
    args = ap.parse_args()
    device = torch.device('cuda')
    model = load_model(args.ckpt, device=device)
    te = CamVid(args.data, 'test')
    res = {}
    for name, c, s in [('clean', None, 0)] + [(f'{c}@2', c, 2) for c in SEEN + HELDOUT]:
        res[name] = analyse(model, FixedCorrupted(te, c, s, cache=args.cache), device)
        print(name, {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)}, flush=True)
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
