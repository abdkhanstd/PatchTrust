"""Zero-shot 11-class evaluation on KITTI semantics (200 labelled images) under
clean, seen and held-out corruptions at severity 2, for a base or PatchTrust checkpoint."""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import KittiSem, NUM_CLASSES
from .corruptions import SEEN, HELDOUT, ALL, apply_corruption
from .models import normalize
from .metrics import per_image_iu, miou_from
from .evaluate import load_model


class Corr(Dataset):
    def __init__(self, b, c, s):
        self.b, self.c, self.s = b, c, s

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        x, y = self.b[i]
        if self.c is None:
            return x, y
        rng = np.random.default_rng([2026, ALL.index(self.c), self.s, i])
        return torch.from_numpy(apply_corruption(x.numpy(), self.c, self.s, rng)), y


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=os.path.expanduser('~/patchtrust/data/kitti_sem'))
    a = ap.parse_args()
    dev = torch.device('cuda')
    m = load_model(a.ckpt, device=dev)
    kd = KittiSem(a.root)
    res = {}
    for c in [None] + SEEN + HELDOUT:
        I, U = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
        for x, y in DataLoader(Corr(kd, c, 2), batch_size=8, num_workers=4):
            out = m(normalize(x.to(dev)))
            i, u, _, _ = per_image_iu(out['final'].argmax(1), y.to(dev), NUM_CLASSES)
            I += i.sum(0).cpu().numpy(); U += u.sum(0).cpu().numpy()
        mi, iou = miou_from(I, U)
        res[c or 'clean'] = {'miou': float(mi), 'iou': [float(v) for v in iou]}
        print(c or 'clean', round(mi, 4), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
