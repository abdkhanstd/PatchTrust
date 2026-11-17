"""Pre-compute the deterministic corrupted test images once (uint8 .npy per
condition) so that every evaluated model sees byte-identical inputs."""
import argparse
import os
import numpy as np
from .data import CamVid, FixedCorrupted, KittiRoad, make_dataset
from .corruptions import ALL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    ap.add_argument('--out', default=os.path.expanduser('~/patchtrust/data/cache'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--dataset', default='camvid', choices=['camvid', 'kitti'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    te = make_dataset(args.dataset, args.data, args.split)
    conds = [(None, 0)] + [(c, s) for c in ALL for s in (1, 2, 3)]
    for c, s in conds:
        sp = te.split
        name = f'{sp}_{c}@{s}' if c else f'{sp}_clean'
        f = os.path.join(args.out, name + '.npy')
        if os.path.exists(f):
            continue
        ds = FixedCorrupted(te, c, s)
        arr = np.stack([ds[i][0].numpy() for i in range(len(ds))])
        np.save(f + '.tmp.npy', arr)
        os.replace(f + '.tmp.npy', f)
        print(name, arr.shape, flush=True)


if __name__ == '__main__':
    main()
