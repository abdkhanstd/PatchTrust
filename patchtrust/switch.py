"""Regression-aware baseline: a confidence switch between the clean base A0 and a
corruption-robust model B (A1 or A1ft). Each pixel keeps A0's label where A0's
max-probability exceeds t and takes B's label elsewhere. Sweeping t traces seen-
corruption mIoU against the clean negative flip rate w.r.t. A0, the trade-off on
which PatchTrust on the frozen A0 is compared."""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CamVid, FixedCorrupted, NUM_CLASSES, IGNORE
from .corruptions import SEEN
from .models import normalize
from .evaluate import load_model
from .coverage import miou_conf

TS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]


@torch.no_grad()
def run(a0, b, ds, device):
    Y, P0, PB, C0 = [], [], [], []
    for x, y in DataLoader(ds, batch_size=16, num_workers=4):
        x, y = normalize(x.to(device)), y.to(device)
        l0 = a0(x)['final']
        v = y != IGNORE
        Y.append(y[v]); P0.append(l0.argmax(1)[v]); C0.append(l0.softmax(1).max(1)[0][v])
        PB.append(b(x)['final'].argmax(1)[v])
    Y, P0, PB, C0 = [torch.cat(t) for t in (Y, P0, PB, C0)]
    out = []
    for t in TS:
        pred = torch.where(C0 > t, P0, PB)
        c0 = P0 == Y
        out.append({'t': t, 'miou': miou_conf(Y, pred), 'nfr': float((c0 & (pred != Y)).sum() / c0.sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a0', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--cache', default=os.path.expanduser('~/patchtrust/data/cache'))
    a = ap.parse_args()
    dev = torch.device('cuda')
    a0, b = load_model(a.a0, device=dev), load_model(a.b, device=dev)
    te = CamVid(a.data, 'test')
    res = {'clean': run(a0, b, FixedCorrupted(te, cache=a.cache), dev)}
    for c in SEEN:
        for s in (1, 2, 3):
            res[f'{c}@{s}'] = run(a0, b, FixedCorrupted(te, c, s, cache=a.cache), dev)
    json.dump(res, open(a.out, 'w'), indent=1)


def matched(files, nfr_target, seen_conds):
    """Seen mIoU and clean mIoU of the switch at a given clean flip rate, by linear
    interpolation on the seed-mean t-sweep."""
    J = [json.load(open(f)) for f in files]
    nfr = np.mean([[e['nfr'] for e in j['clean']] for j in J], 0)
    seen = np.mean([[np.mean([j[c][i]['miou'] for c in seen_conds]) for i in range(len(TS))] for j in J], 0)
    clean = np.mean([[e['miou'] for e in j['clean']] for j in J], 0)
    o = np.argsort(nfr)
    return float(np.interp(nfr_target, nfr[o], seen[o])), float(np.interp(nfr_target, nfr[o], clean[o]))


def summary():
    """python -c 'from patchtrust.switch import summary; summary()' reproduces the paper numbers."""
    import glob
    seen = [f'{c}@{s}' for c in SEEN for s in (1, 2, 3)]
    for b in ['A1', 'A1ft']:
        print(b, matched(sorted(glob.glob(f'results/switch/{b}_s[0-9].json')), 0.0114, seen))


if __name__ == '__main__':
    main()
