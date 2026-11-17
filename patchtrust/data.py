"""CamVid (SegNet 11-class split 367/101/233) and KITTI Road loaders.

Images are kept as uint8 HWC numpy arrays so that corruptions act on the
camera-like [0,255] domain. Normalisation happens on the device (see models.py).
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .corruptions import SEEN, apply_corruption

CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol',
           'Fence', 'Car', 'Pedestrian', 'Bicyclist']
NUM_CLASSES = 11
IGNORE = 255
SIZE = (256, 352)  # H, W


def _read_split(root, split):
    names = sorted(os.listdir(os.path.join(root, split)))
    imgs, labs = [], []
    for n in names:
        im = cv2.imread(os.path.join(root, split, n), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        lb = cv2.imread(os.path.join(root, split + 'annot', n), cv2.IMREAD_UNCHANGED)
        assert im.shape[:2] == lb.shape[:2], n
        lb = lb.astype(np.uint8)
        lb[lb >= NUM_CLASSES] = IGNORE  # class 11 is 'void'
        imgs.append(im)
        labs.append(lb)
    return names, imgs, labs


class CamVid(Dataset):
    """Returns (clean uint8 HWC, corrupted uint8 HWC, label HW int64, corruption id).

    train=True: random scale [0.75,1.5], random crop to SIZE, horizontal flip.
    corrupt_p: probability that the second view is corrupted with a random SEEN
    corruption at random severity 1..3 (else the second view equals the clean one).
    """

    def __init__(self, root, split, train=False, corrupt_p=0.0, subset=None, seed=0):
        self.split = split
        self.names, self.imgs, self.labs = _read_split(root, split)
        if subset is not None:
            rng = np.random.RandomState(seed)
            idx = np.sort(rng.choice(len(self.names), subset, replace=False))
            self.names = [self.names[i] for i in idx]
            self.imgs = [self.imgs[i] for i in idx]
            self.labs = [self.labs[i] for i in idx]
        self.train = train
        self.corrupt_p = corrupt_p
        self.base_size = SIZE  # resize target at evaluation and reference size of the random scale

    def __len__(self):
        return len(self.names)

    def _resize(self, im, lb, h, w):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        lb = cv2.resize(lb, (w, h), interpolation=cv2.INTER_NEAREST)
        return im, lb

    def __getitem__(self, i):
        im, lb = self.imgs[i], self.labs[i]
        H, W = SIZE
        BH, BW = self.base_size
        if self.train:
            rng = np.random.default_rng()
            s = rng.uniform(0.75, 1.5)
            h, w = int(round(BH * s)), int(round(BW * s))
            im, lb = self._resize(im, lb, h, w)
            ph, pw = max(H - h, 0), max(W - w, 0)
            if ph or pw:
                im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                lb = cv2.copyMakeBorder(lb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE)
            y0 = rng.integers(0, im.shape[0] - H + 1)
            x0 = rng.integers(0, im.shape[1] - W + 1)
            im, lb = im[y0:y0 + H, x0:x0 + W], lb[y0:y0 + H, x0:x0 + W]
            if rng.random() < 0.5:
                im, lb = im[:, ::-1], lb[:, ::-1]
            im, lb = np.ascontiguousarray(im), np.ascontiguousarray(lb)
            cid = -1
            corr = im
            if rng.random() < self.corrupt_p:
                cid = int(rng.integers(len(SEEN)))
                sev = int(rng.integers(1, 4))
                corr = apply_corruption(im, SEEN[cid], sev, rng)
        else:
            im, lb = self._resize(im, lb, BH, BW)
            corr, cid = im, -1
        return (torch.from_numpy(im.copy()), torch.from_numpy(np.ascontiguousarray(corr)),
                torch.from_numpy(lb.astype(np.int64)), cid)


def cache_split(base):
    return getattr(base, 'split', 'test')


class FixedCorrupted(Dataset):
    """Deterministic evaluation set: every image corrupted with one (type, severity)
    using a per-image seed, so all methods see identical inputs."""

    def __init__(self, base: CamVid, ctype=None, severity=0, seed=2026, cache=None):
        self.base, self.ctype, self.sev, self.seed = base, ctype, severity, seed
        self.arr = None
        if cache is not None:  # pre-computed images from cache_corruptions.py (identical bytes)
            f = os.path.join(cache, f"{cache_split(base)}_{ctype}@{severity}.npy" if ctype else f"{cache_split(base)}_clean.npy")
            if os.path.exists(f):
                self.arr = np.load(f, mmap_mode='r')
                assert len(self.arr) == len(base)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        im, _, lb, _ = self.base[i]
        if self.arr is not None:
            return torch.from_numpy(np.array(self.arr[i])), lb
        im = im.numpy()
        if self.ctype is not None:
            from .corruptions import ALL
            rng = np.random.default_rng([self.seed, ALL.index(self.ctype), self.sev, i])
            im = apply_corruption(im, self.ctype, self.sev, rng)
        return torch.from_numpy(np.ascontiguousarray(im)), lb


class KittiRoad(Dataset):
    """KITTI Road training split with official road ground truth (289 images).
    Road = magenta (255,0,255). Pixels with blue channel 0 and red 0 (black) are
    outside the evaluated area and ignored. Images resized to height 256, width 848."""

    def __init__(self, root, size=(256, 848)):
        gdir = os.path.join(root, 'training', 'gt_image_2')
        self.items = []
        for g in sorted(os.listdir(gdir)):
            if '_road_' not in g:
                continue
            img = g.replace('_road_', '_')
            self.items.append((os.path.join(root, 'training', 'image_2', img), os.path.join(gdir, g)))
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, gp = self.items[i]
        H, W = self.size
        im = cv2.imread(ip, cv2.IMREAD_COLOR)[:, :, ::-1]
        gt = cv2.imread(gp, cv2.IMREAD_COLOR)[:, :, ::-1]
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        lab = np.full((H, W), IGNORE, np.uint8)
        lab[(gt[:, :, 2] > 127) & (gt[:, :, 0] > 127)] = 1  # magenta road
        lab[(gt[:, :, 0] > 127) & (gt[:, :, 2] <= 127)] = 0  # red non-road
        return torch.from_numpy(np.ascontiguousarray(im)), torch.from_numpy(lab.astype(np.int64))


# Cityscapes labelIds (KITTI semantics) -> CamVid 11 classes; unmapped ids are ignored.
CS2CAMVID = {7: 3, 8: 4, 11: 1, 12: 1, 13: 7, 17: 2, 18: 2, 19: 6, 20: 6, 21: 5, 23: 0, 24: 9, 25: 10,
             26: 8, 27: 8, 28: 8, 29: 8, 30: 8, 31: 8, 32: 10, 33: 10}


class KittiSem(Dataset):
    """KITTI semantic segmentation training images (200, Cityscapes label ids),
    mapped to the 11 CamVid classes for zero-shot evaluation. Resized to 256x848."""

    def __init__(self, root, size=(256, 848)):
        self.root, self.size = root, size
        self.names = sorted(os.listdir(os.path.join(root, 'training', 'semantic')))
        self.lut = np.full(256, IGNORE, np.uint8)
        for k, v in CS2CAMVID.items():
            self.lut[k] = v

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        H, W = self.size
        im = cv2.imread(os.path.join(self.root, 'training', 'image_2', n), cv2.IMREAD_COLOR)[:, :, ::-1]
        lb = cv2.imread(os.path.join(self.root, 'training', 'semantic', n), cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        lb = cv2.resize(self.lut[lb], (W, H), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.ascontiguousarray(im)), torch.from_numpy(lb.astype(np.int64))


class KittiSemTrain(CamVid):
    """KITTI semantics (200 labelled scenes) as a second training dataset with the
    11 CamVid classes. Fixed split by sorted index: test i%4==0 (50), validation
    i%8==1 (25), training the remaining 125. Images are stored at 256x848; training
    uses the same random scale and 256x352 crops as CamVid, evaluation the full 256x848."""

    def __init__(self, root, split, train=False, corrupt_p=0.0, subset=None, seed=0):
        self.split = 'kitti_' + split
        names = sorted(os.listdir(os.path.join(root, 'training', 'semantic')))
        idx = [i for i in range(len(names)) if
               {'test': i % 4 == 0, 'val': i % 8 == 1, 'train': i % 4 != 0 and i % 8 != 1}[split]]
        lut = np.full(256, IGNORE, np.uint8)
        for k, v in CS2CAMVID.items():
            lut[k] = v
        self.names, self.imgs, self.labs = [], [], []
        for i in idx:
            n = names[i]
            im = cv2.imread(os.path.join(root, 'training', 'image_2', n), cv2.IMREAD_COLOR)[:, :, ::-1]
            lb = lut[cv2.imread(os.path.join(root, 'training', 'semantic', n), cv2.IMREAD_UNCHANGED)]
            self.names.append(n)
            self.imgs.append(np.ascontiguousarray(cv2.resize(im, (848, 256), interpolation=cv2.INTER_LINEAR)))
            self.labs.append(cv2.resize(lb, (848, 256), interpolation=cv2.INTER_NEAREST))
        self.train, self.corrupt_p = train, corrupt_p
        self.base_size = (256, 848)


def make_dataset(name, root, split, **kw):
    return KittiSemTrain(root, split, **kw) if name == 'kitti' else CamVid(root, split, **kw)
