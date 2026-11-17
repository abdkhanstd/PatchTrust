"""Budget-swept decomposition of the remaining headroom. At each budget k (share of
valid pixels allowed to change), compare
  learned  : top-k pixels by low r, learned correction L_b + dL
  det_perf : top-k pixels by low r, perfect correction (true label)
  orc_corr : top-k pixels from the true base-error mask (oracle detection; if k
             exceeds the error rate the remaining budget is filled by low r),
             learned correction L_b + dL
det_perf holds the detector fixed and makes the correction perfect (it is limited only
by how many base errors the ranking finds); orc_corr makes the detection perfect and
keeps the learned correction (it is limited only by how many given errors the corrector
fixes). Note the asymmetry: the perfect corrector uses the label, the perfect detector
only the error indicator."""
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from .data import CamVid, FixedCorrupted, IGNORE
from .corruptions import SEEN, HELDOUT
from .models import normalize
from .evaluate import load_model
from .coverage import miou_conf

KS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


@torch.no_grad()
def gather(model, ds, device):
    Y, PB, PU, R = [], [], [], []
    for x, y in DataLoader(ds, batch_size=16, num_workers=4):
        x, y = normalize(x.to(device)), y.to(device)
        out = model(x)
        v = y != IGNORE
        Y.append(y[v]); PB.append(out['base'].argmax(1)[v])
        PU.append((out['base'] + out['dl']).argmax(1)[v]); R.append(out['r'].squeeze(1)[v])
    return [torch.cat(t) for t in (Y, PB, PU, R)]


def sweep(Y, PB, PU, R):
    n = Y.numel()
    wrong = (PB != Y)
    by_r = torch.argsort(R)                                  # least reliable first
    orc = torch.argsort(wrong.float() * 2 - R, descending=True)  # true errors first, then low r
    res = {'err_rate': float(wrong.float().mean())}
    for k in KS:
        m = int(round(k * n))
        sel_r = torch.zeros(n, dtype=torch.bool, device=Y.device); sel_r[by_r[:m]] = True
        sel_o = torch.zeros(n, dtype=torch.bool, device=Y.device); sel_o[orc[:m]] = True
        res[str(k)] = {
            'learned': miou_conf(Y, torch.where(sel_r, PU, PB)),
            'det_perf': miou_conf(Y, torch.where(sel_r, Y, PB)),
            'orc_corr': miou_conf(Y, torch.where(sel_o, PU, PB)),
            'det_recall': float((sel_r & wrong).sum() / wrong.sum().clamp_min(1)),
        }
    res['base'] = miou_conf(Y, PB)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--cache', default=os.path.expanduser('~/patchtrust/data/cache'))
    a = ap.parse_args()
    dev = torch.device('cuda')
    m = load_model(a.ckpt, device=dev)
    te = CamVid(a.data, 'test')
    res = {}
    for c in SEEN + HELDOUT:
        res[f'{c}@2'] = sweep(*gather(m, FixedCorrupted(te, c, 2, cache=a.cache), dev))
        print(c, {k: (round(v['det_perf'], 3), round(v['orc_corr'], 3)) for k, v in res[f'{c}@2'].items() if isinstance(v, dict)}, flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
