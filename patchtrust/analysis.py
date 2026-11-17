"""Aggregate evaluation JSON/NPZ files over seeds, compute paired statistics and
write results tables (CSV/JSON) used for the paper.

Usage: python -m patchtrust.analysis --eval_dir ~/patchtrust/eval --out results/summary.json
"""
import argparse
import glob
import json
import os
import re
import numpy as np
from scipy import stats

from .corruptions import SEEN, HELDOUT

CLS = {'road': 3, 'car': 8, 'ped': 9}


def cond_names(kind, sevs=(1, 2, 3)):
    src = SEEN if kind == 'seen' else HELDOUT
    return [f'{c}@{s}' for c in src for s in sevs]


def summarize(r):
    out = {'clean': r['clean']['miou']}
    for kind in ['seen', 'heldout']:
        cs = cond_names(kind)
        out[kind] = float(np.mean([r[c]['miou'] for c in cs]))
        for s in (1, 2, 3):
            out[f'{kind}_s{s}'] = float(np.mean([r[c]['miou'] for c in cond_names(kind, (s,))]))
        for k, i in CLS.items():
            out[f'{kind}_{k}'] = float(np.mean([r[c]['iou'][i] for c in cs]))
        out[f'{kind}_bf1'] = float(np.mean([r[c]['bf1'] for c in cs]))
        for m in ['ece', 'brier', 'nll']:
            out[f'{kind}_{m}'] = float(np.mean([r[c]['msp'][m] for c in cs]))
            out[f'{kind}_{m}_ts'] = float(np.mean([r[c]['msp_ts'][m] for c in cs]))
        out[f'{kind}_aurc_msp'] = float(np.mean([r[c]['msp']['aurc'] for c in cs]))
        if 'q' in r[cs[0]]:
            out[f'{kind}_aurc_q'] = float(np.mean([r[c]['q']['aurc'] for c in cs]))
            out[f'{kind}_ece_q'] = float(np.mean([r[c]['q']['ece'] for c in cs]))
        # negative / positive flips relative to the reference (clean A0) model
        rf = np.sum([r[c]['ref_flips'] for c in cs], 0)
        if rf[0] > 0:
            out[f'{kind}_nfr'] = float(rf[1] / rf[0])
            out[f'{kind}_pfr'] = float(rf[3] / max(rf[2], 1))
        g = np.sum([r[c]['gstat'] for c in cs], 0)
        if g[1] > 0:
            out[f'{kind}_g_correct'] = float(g[0] / g[1])
            out[f'{kind}_g_wrong'] = float(g[2] / max(g[3], 1))
        f = np.sum([r[c]['flips'] for c in cs], 0)
        out[f'{kind}_selfnfr'] = float(f[1] / max(f[0], 1))
    for k, i in CLS.items():
        out[f'clean_{k}'] = r['clean']['iou'][i]
    out['clean_bf1'] = r['clean']['bf1']
    out['clean_ece'] = r['clean']['msp']['ece']
    out['clean_brier'] = r['clean']['msp']['brier']
    out['clean_ece_ts'] = r['clean']['msp_ts']['ece']
    out['clean_aurc_msp'] = r['clean']['msp']['aurc']
    if 'q' in r['clean']:
        out['clean_aurc_q'] = r['clean']['q']['aurc']
        out['clean_ece_q'] = r['clean']['q']['ece']
    rf = r['clean']['ref_flips']
    if rf[0] > 0:
        out['clean_nfr'] = rf[1] / rf[0]
        out['clean_pfr'] = rf[3] / max(rf[2], 1)
    if 'r_vs_base' in r['clean']:
        out['clean_r_auroc'] = r['clean']['r_vs_base']['auroc']
        out['seen_r_auroc'] = float(np.mean([r[c]['r_vs_base']['auroc'] for c in cond_names('seen')]))
    if 'kitti' in r:
        out['kitti_clean'] = r['kitti']['clean']['road_iou']
        out['kitti_seen'] = float(np.mean([r['kitti'][c]['road_iou'] for c in SEEN]))
        out['kitti_heldout'] = float(np.mean([r['kitti'][c]['road_iou'] for c in HELDOUT]))
    return out


def per_image_miou_matrix(npz, conds):
    """Mean over conditions of dataset-level IoU, returned as I,U arrays (n_img, nc)
    summed over the given conditions for image-level bootstrap."""
    I = sum(npz[f'{c}|I'].astype(np.float64) for c in conds)
    U = sum(npz[f'{c}|U'].astype(np.float64) for c in conds)
    return I, U


def miou_IU(I, U):
    i, u = I.sum(0), U.sum(0)
    return np.nanmean(np.where(u > 0, i / np.maximum(u, 1), np.nan))


def paired_bootstrap(npzA, npzB, conds, B=2000, seed=0):
    """Image-level paired bootstrap of the mIoU difference A-B (pooled over seeds
    and the given conditions). Returns mean diff, 95% CI and one-sided p (A<=B)."""
    rng = np.random.default_rng(seed)
    per = []
    for a, b in zip(npzA, npzB):
        for c in conds:
            per.append((a[f'{c}|I'], a[f'{c}|U'], b[f'{c}|I'], b[f'{c}|U']))
    n = per[0][0].shape[0]
    def diff(idx):
        return np.mean([miou_IU(ia[idx], ua[idx]) - miou_IU(ib[idx], ub[idx]) for ia, ua, ib, ub in per])
    d0 = diff(np.arange(n))
    ds = np.array([diff(rng.integers(0, n, n)) for _ in range(B)])
    return float(d0), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], float(np.mean(ds <= 0))


def hierarchical_bootstrap(npzA, npzB, conds, B=1000, seed=0):
    """Two-level bootstrap of the mean mIoU difference A-B: resample seeds with
    replacement, then test images within each resampled seed. Also returns the
    per-seed differences (mean over conditions)."""
    rng = np.random.default_rng(seed)
    S = len(npzA)
    n = npzA[0][f'{conds[0]}|I'].shape[0]
    def d(s, idx):
        a, b = npzA[s], npzB[s]
        return np.mean([miou_IU(a[f'{c}|I'][idx], a[f'{c}|U'][idx]) - miou_IU(b[f'{c}|I'][idx], b[f'{c}|U'][idx]) for c in conds])
    full = np.arange(n)
    per_seed = [float(d(s, full)) for s in range(S)]
    ds = []
    for _ in range(B):
        ss = rng.integers(0, S, S)
        ds.append(np.mean([d(s, rng.integers(0, n, n)) for s in ss]))
    ds = np.array(ds)
    return float(np.mean(per_seed)), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', default=os.path.expanduser('~/patchtrust/eval'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', default='', help='comma list A:B for paired tests')
    ap.add_argument('--B', type=int, default=500)
    ap.add_argument('--data', default=os.path.expanduser('~/patchtrust/data/CamVid'))
    args = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(args.eval_dir, '*.json'))):
        name = os.path.basename(f)[:-5]
        m = re.match(r'(.*)_s(\d+)$', name)
        if not m:
            continue
        runs.setdefault(m.group(1), {})[int(m.group(2))] = f
    table = {}
    def with_subset(f):
        d = summarize(json.load(open(f)))
        npz = f[:-5] + '.npz'
        if os.path.exists(npz) and not os.path.basename(f).startswith('K'):  # Seq05VD-only (CamVid runs only) test images (indices 62..232): a sequence never seen in training
            z = np.load(npz)
            names = sorted(os.listdir(os.path.join(args.data, 'test')))
            sub = np.array([i for i, nm in enumerate(names) if nm.startswith('Seq05VD')])
            for kind, conds in [('clean', ['clean']), ('seen', cond_names('seen')), ('heldout', cond_names('heldout'))]:
                d[f'seq05_{kind}'] = float(np.mean([miou_IU(z[f'{c}|I'][sub], z[f'{c}|U'][sub]) for c in conds]))
        return d
    for meth, seeds in runs.items():
        S = [with_subset(seeds[s]) for s in sorted(seeds)]
        keys = S[0].keys()
        table[meth] = {'n_seeds': len(S)}
        for k in keys:
            v = np.array([x[k] for x in S if k in x], float)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            ci = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
            table[meth][k] = {'mean': float(v.mean()), 'std': float(sd), 'ci95': float(ci), 'vals': v.tolist()}
    tests = {}
    for pr in [p for p in args.pairs.split(',') if p]:
        A, B = pr.split(':')
        common = sorted(set(runs.get(A, {})) & set(runs.get(B, {})))
        if not common:
            continue
        nA = [np.load(runs[A][s][:-5] + '.npz') for s in common]
        nB = [np.load(runs[B][s][:-5] + '.npz') for s in common]
        tests[pr] = {}
        for kind, conds in [('clean', ['clean']), ('seen', cond_names('seen')), ('heldout', cond_names('heldout'))]:
            d, ci, p = paired_bootstrap(nA, nB, conds, B=args.B)
            # seed-and-condition-level paired Wilcoxon on mIoU
            ja = [json.load(open(runs[A][s])) for s in common]
            jb = [json.load(open(runs[B][s])) for s in common]
            xa = [j[c]['miou'] for j in ja for c in conds]
            xb = [j[c]['miou'] for j in jb for c in conds]
            w = stats.wilcoxon(xa, xb).pvalue if len(xa) > 5 and np.any(np.array(xa) != np.array(xb)) else float('nan')
            hd, hci, ps = hierarchical_bootstrap(nA, nB, conds, B=args.B)
            tests[pr][kind] = {'diff': d, 'ci95': ci, 'p_boot_le0': p, 'wilcoxon_p': float(w), 'n_pairs': len(xa),
                               'hier_ci95': hci, 'per_seed': ps}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump({'table': table, 'tests': tests}, open(args.out, 'w'), indent=1)
    for meth in sorted(table):
        t = table[meth]
        g = lambda k: f"{100*t[k]['mean']:.2f}±{100*t[k]['std']:.2f}" if k in t else '-'
        print(f"{meth:14s} clean {g('clean')} seen {g('seen')} held {g('heldout')} nfr_seen {g('seen_nfr')} ece_seen {g('seen_ece')} aurc_q {g('seen_aurc_q')} kitti {g('kitti_clean')}")
    for pr, d in tests.items():
        print(pr, {k: (round(100 * v['diff'], 2), [round(100 * c, 2) for c in v['hier_ci95']], [round(100 * c, 2) for c in v['per_seed']]) for k, v in d.items()})


if __name__ == '__main__':
    main()
