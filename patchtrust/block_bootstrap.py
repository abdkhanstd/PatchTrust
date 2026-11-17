"""Sequence-aware check of the CamVid confidence intervals: two-level bootstrap
(seeds, then moving blocks of L consecutive test frames within each test sequence,
0001TP = indices 0..61, Seq05VD = 62..232 in sorted file order).
Run: python -m patchtrust.block_bootstrap  (writes results/block_bootstrap.json)"""
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'eval', '')
SEEN = ['gaussian_noise', 'gaussian_blur', 'low_light', 'jpeg', 'occlusion', 'local_blur']
HELD = ['motion_blur', 'contrast', 'fog', 'pixelate']
CONDS = {'clean': ['clean'], 'seen': [f'{c}@{s}' for c in SEEN for s in (1, 2, 3)],
         'heldout': [f'{c}@{s}' for c in HELD for s in (1, 2, 3)]}
SEQS = [np.arange(0, 62), np.arange(62, 233)]  # 0001TP, Seq05VD in sorted file order


def miou(I, U):
    i, u = I.sum(0), U.sum(0)
    return np.nanmean(np.where(u > 0, i / np.maximum(u, 1), np.nan))


def load(run):
    return [dict(np.load(R + f'{run}_s{s}.npz')) for s in range(3)]


def d(a, b, conds, idx):
    return np.mean([miou(a[f'{c}|I'][idx], a[f'{c}|U'][idx]) - miou(b[f'{c}|I'][idx], b[f'{c}|U'][idx]) for c in conds])


def blocks_idx(rng, L):
    out = []
    for seq in SEQS:  # moving-block bootstrap within each sequence, same length
        n = len(seq)
        nb = int(np.ceil(n / L))
        st = rng.integers(0, n - L + 1, nb)
        out.append(np.concatenate([seq[s:s + L] for s in st])[:n])
    return np.concatenate(out)


def run(A, B, L, nB=500, seed=0):
    a, b = load(A), load(B)
    rng = np.random.default_rng(seed)
    res = {}
    for k, conds in CONDS.items():
        full = np.arange(233)
        ps = [d(a[s], b[s], conds, full) for s in range(3)]
        ds = []
        for _ in range(nB):
            ss = rng.integers(0, 3, 3)
            ds.append(np.mean([d(a[s], b[s], conds, blocks_idx(rng, L)) for s in ss]))
        res[k] = (round(100 * np.mean(ps), 2), [round(100 * np.percentile(ds, q), 2) for q in (2.5, 97.5)])
    return res


if __name__ == '__main__':
    import json
    out = {}
    for L in (10, 20):
        for A, B in [('A1_A5', 'A1'), ('A1_A5', 'A1cont'), ('A1_A5', 'A1dec'), ('A0_A5', 'A0'), ('S1_A5', 'S1')]:
            r = run(A, B, L)
            print(L, A, B, r, flush=True)
            out[f'block{L}|{A}:{B}'] = {k: {'diff_pp': float(v[0]), 'ci95_pp': [float(x) for x in v[1]]} for k, v in r.items()}
    path = os.path.join(os.path.dirname(R.rstrip('/\\')), 'block_bootstrap.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)
