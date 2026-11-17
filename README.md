# PatchTrust

Code, results and model weights for **PatchTrust: Spatial Reliability-Guided Learning for Robust Lightweight Driving Scene Segmentation**.

PatchTrust is a 0.11 M-parameter plug-in for a *frozen* LR-ASPP MobileNetV3 segmenter. A reliability head predicts, per pixel, the probability `r` that the frozen prediction is correct (and `q` for the final prediction). A residual head proposes class-logit corrections `dL`, and the final logits are `L = L_base + (1 - r) * dL`. Training uses paired clean and corrupted crops and a confidence-filtered clean-image teacher. No extra labels are needed.

## Main results (CamVid test, mean ± std over 3 seeds, mIoU %)

| Method | Trained params | Clean | Seen corruptions | Held-out corruptions |
|---|---|---|---|---|
| Clean base A0 | 3.22 M | 57.4 ± 0.3 | 41.6 ± 0.7 | 42.8 ± 1.1 |
| Corruption-augmented A1 | 3.22 M | 56.1 ± 0.8 | 51.5 ± 0.4 | 46.6 ± 0.6 |
| AugMix-style A1jsd (run name `A1c`) | 3.22 M | 51.6 ± 0.5 | 48.4 ± 0.7 | 44.1 ± 0.5 |
| A1 + 80 epochs (A1cont) | 3.22 M | 55.9 ± 0.6 | 51.4 ± 0.0 | 46.6 ± 0.7 |
| PatchTrust on frozen A0 | 0.11 M | 57.0 ± 0.4 | 48.0 ± 0.8 | 44.5 ± 0.2 |
| A1 + decoder fine-tune (A1dec) | 0.25 M | 56.1 ± 0.7 | 51.6 ± 0.3 | 46.7 ± 0.4 |
| A1 + TENT (test-time, batches of 8) | 0 | 52.9 ± 0.6 | 47.6 ± 0.7 | **48.6 ± 0.7** |
| PatchTrust on frozen A1 | 0.11 M | 56.8 ± 0.6 | **52.1 ± 0.3** | 47.5 ± 0.5 |
| MobileNetV3-Small S1 (corruption-aug.) | 1.08 M | 51.4 ± 0.4 | 47.4 ± 0.6 | 42.5 ± 1.1 |
| PatchTrust on frozen S1 | 0.08 M | 52.7 ± 0.7 | 48.4 ± 0.7 | 43.8 ± 1.1 |

On KITTI semantics as a second training set (125/25/50 split), PatchTrust adds only +0.2/+0.3/+0.3 to the corruption-augmented base and continued full training is better; on clean-trained bases it recovers about 6 pp under seen corruptions on both datasets. Test-time BN adaptation with a source prior (`patchtrust/tta.py --mode bnprior`) is best on held-out corruptions. See the paper for the full, mixed picture.

All numbers are produced by `python -m patchtrust.analysis` from the files in `results/eval`. The ablations show that at full coverage the gate type barely matters (the corrector gates itself), while the learned reliability ranks correction sites better than entropy or max-probability on the clean-trained base. See `results/results_summary.json` and `results/coverage/`.

## Layout

```
patchtrust/            package
  data.py              CamVid (SegNet 367/101/233 split) and KITTI Road loaders
  corruptions.py       6 seen + 4 held-out corruptions, 3 severities, deterministic
  models.py            LR-ASPP MobileNetV3 base, PatchTrust heads and gates
  train.py             base training (clean / aug / augjsd) and head training
  evaluate.py          test evaluation, calibration, flips, KITTI zero-shot
  coverage.py          where-to-correct analysis (selectors at fixed budgets)
  analysis.py          aggregation over seeds, per-seed paired differences, seed x image bootstrap (B=500)
  block_bootstrap.py   sequence-aware check: seed x moving-block bootstrap (blocks of 10 or 20 consecutive test frames per sequence)
  profile.py           params, MACs, latency, memory
  cache_corruptions.py pre-computes the deterministic corrupted test sets
  dump_examples.py     qualitative examples
  tta.py               test-time adaptation (BN statistics, BN with source prior, TENT; also PatchTrust on an adapted base)
  mechanism.py         self-gating, detection/correction decomposition, within-image AUROC
  kitti_sem_eval.py    zero-shot 11-class evaluation on KITTI semantics
  decomp.py            budget-matched detection vs correction decomposition
tests/                 unit tests (pytest)
scripts/               job lists and the resumable queue runner used for all runs
figures/               plotting script for the result figures
results/               evaluation outputs (JSON summaries + raw per-image counts), coverage curves, profiles
weights/               all checkpoints (best.pt) and training summaries, one folder per run
logs/                  training and evaluation logs
```

Run names: `A0` clean base, `A1` corruption-augmented base, `A1c` AugMix-style JSD base (called A1jsd in the paper), `A1ft` A0 fine-tuned with corruptions, `A1cont` A1 trained 80 more epochs. `<base>_<variant>` are PatchTrust heads on a frozen base: `A2` no gate, `A3` entropy gate, `A4` no consistency, `A5` full PatchTrust, `A5nc` no cues, `A6` learned gate without reliability supervision, `A5sel` learned arbiter, `A5oracle` privileged oracle gate (evaluation only). `_s0/_s1/_s2` are seeds. `A2nop`/`A5nop` remove the base softmax from the corrector input, `A5cue` trains the reliability head on cues only. `A0dec`/`A1dec`/`S1dec`/`K1dec` fine-tune only the LR-ASPP decoder, `S1cont`/`K1cont` continue training. `K*` runs are trained on KITTI semantics (`--dataset kitti`, split 125/25/50 by sorted index). `S0`/`S1` are MobileNetV3-Small bases (clean / corruption-augmented) and `S*_A5`, `S*_A2` their heads. `A0_A5tf_s0` and `A0_A5gg_s0` are seed-0 pilots (teacher-forced corrector, gate gradient).

## Setup

Python 3.10, PyTorch 2.1.2, torchvision 0.16.2 (CUDA 11.8 tested on RTX 2080 Ti), see `requirements.txt`.

```bash
pip install -r requirements.txt
```

Data:
* CamVid, SegNet split: `CamVid/` folder of https://github.com/alexgkendall/SegNet-Tutorial (train/val/test and *annot folders, 11 classes, class 11 = void).
* KITTI semantics (optional): `data_semantics.zip` from the KITTI semantic segmentation benchmark; the 200 labelled training images are used zero-shot with Cityscapes ids mapped to the CamVid classes (`patchtrust/data.py`, `CS2CAMVID`).
* KITTI Road (optional): `data_road.zip` from the KITTI Road benchmark page (https://www.cvlibs.net/datasets/kitti/eval_road.php). Only the labelled training images are used, zero-shot.

## Reproduce

```bash
# unit tests
CAMVID=/path/to/CamVid python -m pytest -q tests
# base models (seed 0)
python -m patchtrust.train --data /path/to/CamVid --out runs/A0_s0 --mode clean --seed 0
python -m patchtrust.train --data /path/to/CamVid --out runs/A1_s0 --mode aug --seed 0
# PatchTrust heads on the frozen base
python -m patchtrust.train --data /path/to/CamVid --out runs/A1_A5_s0 --mode heads --base runs/A1_s0/best.pt \
    --gate learned --epochs 80 --lr 1e-3 --eval_every 4 --patience 24 --seed 0
# cached corrupted test sets, evaluation, analysis
python -m patchtrust.cache_corruptions --data /path/to/CamVid --out cache
python -m patchtrust.evaluate --ckpt runs/A1_A5_s0/best.pt --ref runs/A0_s0/best.pt --data /path/to/CamVid \
    --kitti /path/to/data_road --cache cache --out eval/A1_A5_s0
python -m patchtrust.coverage --ckpt runs/A1_A5_s0/best.pt --data /path/to/CamVid --cache cache --out coverage/A1_A5_s0.json
python -m patchtrust.analysis --eval_dir results/eval --data /path/to/CamVid --out summary.json --B 500 \
    --pairs A1_A5:A1,A1_A5:A1cont,A1_A5:A1dec,S1_A5:S1,S1_A5:S1cont,S1_A5:S1dec,K1_A5:K1,K1_A5:K1cont,K1_A5:K1dec,A0_A5:A0,K0_A5:K0
# sequence-aware block bootstrap of the main CamVid comparisons (uses results/eval/*.npz)
python -m patchtrust.block_bootstrap
```

## Where the numbers in the paper come from

* Table means and standard deviations: `results/results_summary.json` -> `table[<run>][<metric>]` (e.g. `clean`, `seen`, `heldout`, `seen_bf1`, `clean_nfr`, `seen_nfr`, `seen_selfnfr`, `seen_brier_ts`, `seen_s2`).
* Paired differences, per-seed values and the seed x image intervals: `results_summary.json` -> `tests['<A>:<B>'][<split>]` (`diff`, `per_seed`, `hier_ci95`).
* Block-bootstrap intervals: `results/block_bootstrap.json`.
* Clean flip rates given only in the text: `table['A0_A5']['clean_nfr']` (1.1 %), `table['K0_A5']['clean_nfr']` (KITTI, 0.7 %), `table['A0_A2']['clean_nfr']` (ungated, 1.1 %).
* TTA, mechanism, coverage, decomposition, KITTI zero-shot, confidence switch and profiling: `results/tta`, `results/mech`, `results/coverage`, `results/decomp`, `results/ksem`, `results/switch`, `results/profile_*.json`.

The full set of runs is listed in `scripts/jobs_*.txt` and was executed with `scripts/run_queue.sh`, which skips finished runs and resumes interrupted ones from `last.pt`. To evaluate the released weights, pass `--ckpt weights/<run>/best.pt` (and `--base weights/<base>/best.pt` for head checkpoints, which store only the head weights).

## Licence

MIT, see `LICENSE`.
