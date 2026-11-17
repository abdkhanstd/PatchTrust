"""Where-to-correct analysis. For a trained PatchTrust model, the candidate
corrected prediction is fixed and only the region where it replaces the base
prediction changes. Pixels are ranked by a selector (learned reliability r,
base entropy, base max-prob, random, or the privileged oracle base-error mask)
and the correction is accepted on the top-k fraction. Reports mIoU, negative
flip rate (NFR) and positive flip rate as functions of the coverage k."""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CamVid, FixedCorrupted, NUM_CLASSES, IGNORE
from .corruptions import SEEN, HELDOUT
from .models import normalize
from .evaluate import load_model

KS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def miou_conf(y, p):
    cm = torch.bincount(y * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES).double()
    I = cm.diag()
    U = cm.sum(0) + cm.sum(1) - I
    iou = torch.where(U > 0, I / U.clamp_min(1), torch.full_like(U, float('nan')))
    return float(torch.nanmean(iou))


@torch.no_grad()
def collect(model, ds, device):
    Y, PB, PC, R, H, M = [], [], [], [], [], []
    for x, y in DataLoader(ds, batch_size=16, num_workers=4):
        x, y = normalize(x.to(device)), y.to(device)
        out = model(x)
        lb = out['base']
        pbp = lb.softmax(1)
        v = y != IGNORE
        Y.append(y[v]); PB.append(lb.argmax(1)[v]); PC.append(out['final'].argmax(1)[v])
        R.append(out['r'].squeeze(1)[v]); M.append(pbp.max(1)[0][v])
        H.append((-(pbp * pbp.clamp_min(1e-8).log()).sum(1))[v])
    return [torch.cat(t) for t in (Y, PB, PC, R, H, M)]


def curves(Y, PB, PC, R, H, M, seed=0):
    g = torch.Generator(device=Y.device).manual_seed(seed)
    n = Y.numel()
    sel_scores = {  # higher score = corrected first
        'learned_r': -R.float(),
        'entropy': H.float(),
        'neg_msp': -M.float(),
        'random': torch.rand(n, generator=g, device=Y.device),
        'oracle': (PB != Y).float() + 1e-3 * torch.rand(n, generator=g, device=Y.device),
    }
    bc = PB == Y
    out = {}
    for name, sc in sel_scores.items():
        order = torch.argsort(sc, descending=True)
        res = []
        for k in KS:
            m = torch.zeros(n, dtype=torch.bool, device=Y.device)
            m[order[:int(round(k * n))]] = True
            pred = torch.where(m, PC, PB)
            corr = pred == Y
            res.append({'k': k, 'miou': miou_conf(Y, pred), 'acc': float(corr.float().mean()),
                        'nfr': float((bc & ~corr).sum() / bc.sum().clamp_min(1)),
                        'pfr': float((~bc & corr).sum() / (~bc).sum().clamp_min(1))})
        out[name] = res
    return out


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
        res[name] = curves(*collect(model, FixedCorrupted(te, c, s, cache=args.cache), device))
        print(name, {k: round(v[4]['miou'], 4) for k, v in res[name].items()}, flush=True)
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
