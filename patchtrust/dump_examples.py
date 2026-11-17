"""Dump qualitative examples (image, corrupted image, GT, base prediction,
PatchTrust prediction, reliability map, error maps) to an .npz for plotting."""
import argparse
import os
import numpy as np
import torch

from .data import CamVid, FixedCorrupted
from .models import normalize
from .evaluate import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--items', default='12:low_light@3,60:occlusion@2,150:gaussian_noise@2,200:local_blur@3,100:fog@2')
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--cache', default=os.path.expanduser('~/patchtrust/data/cache'))
    args = ap.parse_args()
    dev = torch.device('cuda')
    m = load_model(args.ckpt, device=dev)
    te = CamVid(args.data, 'test')
    out = {}
    for j, it in enumerate(args.items.split(',')):
        idx, cond = it.split(':')
        c, s = cond.split('@')
        idx = int(idx)
        clean = FixedCorrupted(te, cache=args.cache)[idx][0].numpy()
        x, y = FixedCorrupted(te, c, int(s), cache=args.cache)[idx]
        with torch.no_grad():
            o = m(normalize(x[None].to(dev)))
        out[f'{j}_clean'] = clean
        out[f'{j}_corr'] = x.numpy()
        out[f'{j}_gt'] = y.numpy()
        out[f'{j}_base'] = o['base'].argmax(1)[0].cpu().numpy()
        out[f'{j}_pt'] = o['final'].argmax(1)[0].cpu().numpy()
        out[f'{j}_r'] = o['r'][0, 0].cpu().numpy()
        out[f'{j}_name'] = np.array(f'{te.names[idx]} {cond}')
    np.savez_compressed(args.out, **out)


if __name__ == '__main__':
    main()
