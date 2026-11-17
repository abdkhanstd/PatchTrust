"""Test-time adaptation baselines, evaluated on the same cached test conditions
as evaluate.py:
  bn       - batch-norm statistics re-estimated on each test batch, no gradients
  bnprior  - BN statistics mixed with the source statistics using a pseudo-count
             N (Schneider et al., NeurIPS 2020): mu = (N mu_s + n mu_t) / (N + n)
  tent     - TENT: entropy minimisation of the BN affine parameters, online over the
             stream of one condition, reset between conditions
All are transductive: they use a batch of test images that share one condition.
If --ckpt is a PatchTrust head checkpoint, the adaptation is applied to the
frozen base inside PatchTrust (bn / bnprior only) and the heads are run on the
adapted features, which tests whether the two are complementary."""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import FixedCorrupted, NUM_CLASSES, make_dataset
from .corruptions import SEEN, HELDOUT
from .models import Base, normalize
from .metrics import per_image_iu, miou_from
from .evaluate import load_model


class PriorBN(nn.Module):
    """Wraps a trained BatchNorm2d; normalises with source statistics mixed with the
    statistics of the current test batch (pseudo-count N; N=0 gives pure batch stats)."""

    def __init__(self, bn, N):
        super().__init__()
        self.bn, self.N = bn, N

    def forward(self, x):
        n = x.shape[0] * x.shape[2] * x.shape[3] / (x.shape[2] * x.shape[3])  # number of images
        mu_t = x.mean((0, 2, 3))
        var_t = x.var((0, 2, 3), unbiased=False)
        w = self.N / (self.N + n)
        mu = w * self.bn.running_mean + (1 - w) * mu_t
        var = w * self.bn.running_var + (1 - w) * var_t
        return F.batch_norm(x, mu, var, self.bn.weight, self.bn.bias, False, 0.0, self.bn.eps)


def wrap_bn(module, N):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, PriorBN(child, N))
        else:
            wrap_bn(child, N)


def configure_tent(model):
    model.train()
    params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    return params


def build(args, device):
    ck = torch.load(args.ckpt, map_location='cpu')
    if ck['args']['mode'] == 'heads':
        assert args.mode in ('bn', 'bnprior')
        m = load_model(args.ckpt, device='cpu')
        wrap_bn(m.base, 0 if args.mode == 'bn' else args.N)
        return m.to(device).eval(), None, True
    model = Base(NUM_CLASSES, pretrained_backbone=False, arch=ck['args'].get('arch', 'large'))
    model.load_state_dict(ck['model'])
    params = None
    if args.mode == 'tent':
        params = configure_tent(model)
    else:
        wrap_bn(model, 0 if args.mode == 'bn' else args.N)
        model.eval()
    return model.to(device), params, False


def run(args, ds, device, ref=None):
    model, params, is_pt = build(args, device)
    flips = np.zeros(2)  # pixels correct under the reference, of which wrong now
    opt = torch.optim.Adam(params, lr=args.lr) if params else None
    I, U = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
    for x, y in DataLoader(ds, batch_size=args.bs, num_workers=4, shuffle=False):
        x, y = normalize(x.to(device)), y.to(device)
        if opt is not None:
            out = model(x)
            ent = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
            opt.zero_grad()
            ent.backward()
            opt.step()
            pred = out.detach().argmax(1)  # online prediction, as in TENT
        else:
            with torch.no_grad():
                out = model(x)
                pred = (out['final'] if is_pt else out).argmax(1)
        i, u, _, _ = per_image_iu(pred, y, NUM_CLASSES)
        I += i.sum(0).cpu().numpy()
        U += u.sum(0).cpu().numpy()
        if ref is not None:
            with torch.no_grad():
                rc = (ref(x)['final'].argmax(1) == y) & (y != 255)
            flips += [rc.sum().item(), (rc & (pred != y)).sum().item()]
    return miou_from(I, U)[0], flips.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mode', choices=['bn', 'bnprior', 'tent'], required=True)
    ap.add_argument('--N', type=float, default=16)
    ap.add_argument('--ref', default=None, help='reference checkpoint for negative-flip rates')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='camvid', choices=['camvid', 'kitti'])
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--cache', default=os.path.expanduser('~/patchtrust/data/cache'))
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device('cuda')
    te = make_dataset(args.dataset, args.data, 'test')
    ref = load_model(args.ref, device=device) if args.ref else None
    res = {}
    for name, c, s in [('clean', None, 0)] + [(f'{c}@{s}', c, s) for c in SEEN + HELDOUT for s in (1, 2, 3)]:
        mi, fl = run(args, FixedCorrupted(te, c, s, cache=args.cache), device, ref)
        res[name] = {'miou': float(mi), 'ref_flips': fl}
        print(name, round(res[name]['miou'], 4), flush=True)
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
