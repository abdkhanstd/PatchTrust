"""Paper figures from the result files in code/results (pulled from the server).
All text in Times New Roman, canvases drawn at printed width."""
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, 'results', 'plots'))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 7,
                     'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.3, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'pdf.fonttype': 42})
SEEN = ['gaussian_noise', 'gaussian_blur', 'low_light', 'jpeg', 'occlusion', 'local_blur']
HELD = ['motion_blur', 'contrast', 'fog', 'pixelate']
NICE = {'gaussian_noise': 'Noise', 'gaussian_blur': 'Blur', 'low_light': 'Low light', 'jpeg': 'JPEG',
        'occlusion': 'Occlusion', 'local_blur': 'Local blur', 'motion_blur': 'Motion*', 'contrast': 'Contrast*',
        'fog': 'Fog*', 'pixelate': 'Pixelate*'}
COL = {'learned_r': '#c0392b', 'entropy': '#2471a3', 'neg_msp': '#7d3c98', 'random': '#7f8c8d', 'oracle': '#27ae60'}
LAB = {'learned_r': 'Learned reliability $r$ (ours)', 'entropy': 'Base entropy', 'neg_msp': 'Base max-prob',
       'random': 'Random', 'oracle': 'Oracle error mask (privileged)'}
SAVE = dict(bbox_inches='tight', pad_inches=0.01)


def load_cov(base):
    fs = sorted(glob.glob(os.path.join(RES, 'coverage', f'{base}_A5_s*.json')))
    return [json.load(open(f)) for f in fs]


def cov_curve(covs, conds, sel, key):
    arr = np.array([[[e[key] for e in c[k][sel]] for k in conds] for c in covs])  # seed, cond, k
    return arr.mean(1).mean(0), arr.mean(1).std(0)


def fig_coverage():
    covs = {'A0': load_cov('A0'), 'A1': load_cov('A1')}
    ks = np.array([e['k'] for e in covs['A0'][0]['clean']['learned_r']]) * 100
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.05))
    panels = [('A0 base, seen', 'A0', [f'{c}@2' for c in SEEN]), ('A0 base, held-out', 'A0', [f'{c}@2' for c in HELD]),
              ('A1 base, seen', 'A1', [f'{c}@2' for c in SEEN])]
    for ax, (title, b, conds) in zip(axs[:3], panels):
        for sel in ['oracle', 'learned_r', 'entropy', 'neg_msp', 'random']:
            m, s_ = cov_curve(covs[b], conds, sel, 'miou')
            ax.plot(ks, 100 * m, color=COL[sel], ls='--' if sel == 'oracle' else '-', label=LAB[sel],
                    marker='o', ms=1.8)
            ax.fill_between(ks, 100 * (m - s_), 100 * (m + s_), color=COL[sel], alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel('Corrected pixels (%)')
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel('mIoU (%)')
    ax = axs[3]
    conds = [f'{c}@2' for c in SEEN]
    for sel in ['oracle', 'learned_r', 'entropy', 'neg_msp', 'random']:
        g, _ = cov_curve(covs['A0'], conds, sel, 'miou')
        n, _ = cov_curve(covs['A0'], conds, sel, 'nfr')
        ax.plot(100 * n, 100 * (g - g[0]), color=COL[sel], ls='--' if sel == 'oracle' else '-', marker='o', ms=1.8)
    ax.set_xlabel('Negative flip rate (%)')
    ax.set_ylabel('mIoU gain (pp)')
    ax.set_title('A0 base, gain vs. flips (seen)')
    ax.grid(alpha=0.25, lw=0.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.8)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(os.path.join(OUT, 'coverage.pdf'), **SAVE)
    plt.close(fig)


def load_eval(name):
    fs = sorted(glob.glob(os.path.join(RES, 'eval', f'{name}_s[0-9].json')))
    return [json.load(open(f)) for f in fs]


METH = [('A0', 'Clean base A0', '#7f8c8d', '-'), ('A1', 'Corruption-aug. A1', '#2471a3', '-'),
        ('A1c', 'AugMix-style A1jsd', '#7d3c98', '-'), ('A0_A5', 'PatchTrust on A0', '#e67e22', '--'),
        ('A1_A5', 'PatchTrust on A1', '#c0392b', '-')]


def fig_severity():
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.05), gridspec_kw={'width_ratios': [1, 1, 2.2]})
    for ax, (title, src) in zip(axs[:2], [('Seen corruptions', SEEN), ('Held-out corruptions', HELD)]):
        for key, lab, col, ls in METH:
            E = load_eval(key)
            if not E:
                continue
            ys = []
            for e in E:
                ys.append([e['clean']['miou']] + [np.mean([e[f'{c}@{s}']['miou'] for c in src]) for s in (1, 2, 3)])
            ys = 100 * np.array(ys)
            ax.errorbar([0, 1, 2, 3], ys.mean(0), yerr=ys.std(0), color=col, ls=ls, label=lab, marker='o', ms=2,
                        capsize=1.5, elinewidth=0.6)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Clean', 'S1', 'S2', 'S3'])
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel('mIoU (%)')
    ax = axs[2]
    conds = SEEN + HELD
    w = 0.16
    for i, (key, lab, col, ls) in enumerate(METH):
        E = load_eval(key)
        if not E:
            continue
        v = 100 * np.array([[np.mean([e[f'{c}@{s}']['miou'] for s in (1, 2, 3)]) for c in conds] for e in E])
        ax.bar(np.arange(len(conds)) + (i - 2) * w, v.mean(0), w, yerr=v.std(0), color=col, label=lab,
               error_kw=dict(lw=0.5, capsize=0.8), alpha=0.9 if ls == '-' else 0.55)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([NICE[c] for c in conds], rotation=45, ha='right')
    ax.set_ylabel('mIoU (%), mean of S1-S3')
    ax.set_title('Per corruption (* held out)')
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(OUT, 'severity.pdf'), **SAVE)
    plt.close(fig)


def risk_cov(n, c):
    n, c = n[::-1].cumsum(), c[::-1].cumsum()
    keep = np.r_[True, np.diff(n) > 0]
    return (n / n[-1])[keep], (1 - c / np.maximum(n, 1))[keep]


def fig_riskcov():
    """(a,b) Detection of base-model errors: learned reliability r vs. base max-prob,
    per condition (severity 2), mean over seeds. (c) Risk-coverage of the final
    prediction under seen corruptions at severity 3 (seed 0, pooled)."""
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.05), gridspec_kw={'width_ratios': [1.25, 1.25, 1]})
    conds = ['clean'] + [f'{c}@2' for c in SEEN + HELD]
    x = np.arange(len(conds))
    for ax, base in zip(axs[:2], ['A0', 'A1']):
        E = load_eval(f'{base}_A5')
        r = 100 * np.array([[e[c]['r_vs_base']['auroc'] for c in conds] for e in E])
        m = 100 * np.array([[e[c]['base_msp']['auroc'] for c in conds] for e in E])
        ax.errorbar(x, r.mean(0), yerr=r.std(0), color='#c0392b', marker='o', ms=2, capsize=1, elinewidth=0.5,
                    label='Learned reliability $r$')
        ax.errorbar(x, m.mean(0), yerr=m.std(0), color='#2471a3', marker='x', ms=2.5, ls=':', capsize=1,
                    elinewidth=0.5, label=f'Base max-prob')
        ax.set_xticks(x)
        ax.set_xticklabels(['Clean'] + [NICE[c] for c in SEEN + HELD], rotation=45, ha='right')
        ax.set_title(f'Detecting errors of base {base} (S2)')
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel('AUROC (%)')
    axs[0].legend(frameon=False, loc='lower right', fontsize=6)
    ax = axs[2]
    centers = (np.linspace(0, 1, 11)[:-1] + 0.05) * 100
    for run, lab, col, ls in [('A0_A2', 'A0, corrector sees $p_b$', '#e67e22', '-'), ('A0_A2nop', 'A0, no $p_b$', '#e67e22', ':'),
                              ('A1_A2', 'A1, corrector sees $p_b$', '#c0392b', '-'), ('A1_A2nop', 'A1, no $p_b$', '#c0392b', ':')]:
        J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'mech', f'{run}_s[0-9].json')))]
        cr = np.array([np.mean([j[f'{c}@2']['change_rate_by_msp'] for c in SEEN], 0) for j in J])
        n = np.array([np.mean([j[f'{c}@2']['pix_by_msp'] for c in SEEN], 0) for j in J]).mean(0)
        k = n > 0
        ax.plot(centers[k], 100 * cr.mean(0)[k], color=col, ls=ls, marker='o', ms=1.8, label=lab)
    ax.set_xlabel('Base max-probability (%)')
    ax.set_ylabel(r'Labels changed by $\Delta L$ (%)')
    ax.set_title('Self-gating of the ungated corrector')
    ax.set_ylim(-3, 112)
    ax.legend(frameon=False, fontsize=5.4, loc='upper right', handlelength=1.6, labelspacing=0.2, borderaxespad=0.2)
    ax.grid(alpha=0.25, lw=0.4)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(OUT, 'reliability.pdf'), **SAVE)
    plt.close(fig)


CAMVID_COLORS = np.array([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [60, 40, 222],
                          [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0],
                          [0, 128, 192], [0, 0, 0]], np.uint8)


def colorize(lab):
    l = lab.copy()
    l[l == 255] = 11
    return CAMVID_COLORS[l]


def fig_qual(rows=None, name='qualitative.pdf'):
    f = os.path.join(RES, 'examples_A1.npz')
    if not os.path.exists(f):
        return
    z = np.load(f)
    n = len([k for k in z.files if k.endswith('_name')])
    if rows is not None:
        n = min(n, rows)
    cols = ['Clean image', 'Corrupted input', 'Ground truth', 'Base A1', 'PatchTrust (A1)', 'Reliability $r$',
            'Error: base / ours']
    fig, axs = plt.subplots(n, 7, figsize=(7.16, 0.8 * n))
    for i in range(n):
        gt = z[f'{i}_gt']
        v = gt != 255
        eb = (z[f'{i}_base'] != gt) & v
        ep = (z[f'{i}_pt'] != gt) & v
        err = np.full(gt.shape + (3,), 255, np.uint8)
        err[eb & ep] = [90, 90, 90]
        err[eb & ~ep] = [39, 174, 96]   # fixed by PatchTrust
        err[~eb & ep] = [192, 57, 43]   # broken by PatchTrust
        ims = [z[f'{i}_clean'], z[f'{i}_corr'], colorize(gt), colorize(z[f'{i}_base']), colorize(z[f'{i}_pt']),
               z[f'{i}_r'], err]
        for j, im in enumerate(ims):
            ax = axs[i, j]
            if j == 5:
                ax.imshow(im, cmap='magma', vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if i == 0:
                ax.set_title(cols[j], fontsize=6.5)
        cond = str(z[f'{i}_name']).split(' ')[1].replace('@', ' S')
        for k, v_ in NICE.items():
            cond = cond.replace(k, v_.replace('*', ''))
        axs[i, 0].set_ylabel(cond, fontsize=6.3)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(os.path.join(OUT, name), dpi=220, **SAVE)
    plt.close(fig)


if __name__ == '__main__':
    fig_coverage()
    fig_severity()
    fig_riskcov()
    fig_qual(rows=3)
    print('figures written to', OUT)
