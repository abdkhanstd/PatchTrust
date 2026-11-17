"""Segmentation, boundary, calibration and selective-prediction metrics.
All statistics are computed on valid (non-ignored) pixels only."""
import numpy as np
import torch
import torch.nn.functional as F

IGNORE = 255
NBINS_ECE = 15
NBINS_RC = 1000


def per_image_iu(pred, lab, nc):
    """pred, lab: (B,H,W) long. Returns intersection and union (B,nc) and
    pixel-correct / valid counts (B,)."""
    valid = lab != IGNORE
    B = pred.shape[0]
    I = torch.zeros(B, nc, dtype=torch.long, device=pred.device)
    U = torch.zeros(B, nc, dtype=torch.long, device=pred.device)
    for k in range(nc):
        pk = (pred == k) & valid
        lk = (lab == k)
        I[:, k] = (pk & lk).flatten(1).sum(1)
        U[:, k] = (pk | lk).flatten(1).sum(1)
    corr = ((pred == lab) & valid).flatten(1).sum(1)
    return I, U, corr, valid.flatten(1).sum(1)


def miou_from(I, U, present_only=True):
    """Dataset-level mIoU from summed I,U (nc,). Classes absent from both
    prediction and label (U==0) are excluded."""
    I, U = np.asarray(I, np.float64), np.asarray(U, np.float64)
    iou = np.where(U > 0, I / np.maximum(U, 1), np.nan)
    return np.nanmean(iou), iou


def _boundary(lab):
    b = torch.zeros_like(lab, dtype=torch.bool)
    b[:, :, 1:] |= lab[:, :, 1:] != lab[:, :, :-1]
    b[:, 1:, :] |= lab[:, 1:, :] != lab[:, :-1, :]
    return b


def boundary_counts(pred, lab, tol=2):
    """Class-agnostic boundary F1 counts with a (2*tol+1) square tolerance.
    Boundaries touching ignored pixels are excluded."""
    valid = lab != IGNORE
    vd = -F.max_pool2d(-valid.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0.5  # eroded valid
    bp = _boundary(pred) & vd
    bg = _boundary(lab) & vd
    k = 2 * tol + 1
    dil = lambda b: F.max_pool2d(b.float().unsqueeze(1), k, 1, tol).squeeze(1) > 0.5
    tp_p = (bp & dil(bg)).sum().item()
    tp_r = (bg & dil(bp)).sum().item()
    return np.array([tp_p, bp.sum().item(), tp_r, bg.sum().item()], np.float64)


def bf1_from(c):
    p = c[0] / max(c[1], 1)
    r = c[2] / max(c[3], 1)
    return 2 * p * r / max(p + r, 1e-12)


class CalibAcc:
    """Accumulates histograms for ECE and risk-coverage (AURC) of one confidence
    score, plus Brier score of the softmax."""

    def __init__(self):
        self.ece_n = np.zeros(NBINS_ECE)
        self.ece_c = np.zeros(NBINS_ECE)
        self.ece_s = np.zeros(NBINS_ECE)
        self.rc_n = np.zeros(NBINS_RC)
        self.rc_c = np.zeros(NBINS_RC)
        self.brier_sum = 0.0
        self.nll_sum = 0.0
        self.n = 0

    def add(self, conf, correct, valid, prob=None, lab=None):
        conf = conf[valid].float().clamp(0, 1)
        cor = correct[valid].float()
        b = (conf * NBINS_ECE).long().clamp(max=NBINS_ECE - 1)
        self.ece_n += torch.bincount(b, minlength=NBINS_ECE).cpu().numpy()
        self.ece_c += torch.bincount(b, weights=cor, minlength=NBINS_ECE).cpu().numpy()
        self.ece_s += torch.bincount(b, weights=conf, minlength=NBINS_ECE).cpu().numpy()
        b2 = (conf * NBINS_RC).long().clamp(max=NBINS_RC - 1)
        self.rc_n += torch.bincount(b2, minlength=NBINS_RC).cpu().numpy()
        self.rc_c += torch.bincount(b2, weights=cor, minlength=NBINS_RC).cpu().numpy()
        if prob is not None:
            v = valid
            l = lab.clone()
            l[~v] = 0
            oh = F.one_hot(l, prob.shape[1]).permute(0, 3, 1, 2).float()
            br = ((prob - oh) ** 2).sum(1)
            self.brier_sum += br[v].sum().item()
            pl = prob.gather(1, l.unsqueeze(1)).squeeze(1)
            self.nll_sum += (-torch.log(pl.clamp_min(1e-8)))[v].sum().item()
        self.n += int(valid.sum().item())

    def ece(self):
        n = self.ece_n.sum()
        m = self.ece_n > 0
        return float(np.sum(np.abs(self.ece_c[m] - self.ece_s[m])) / n)

    def brier(self):
        return self.brier_sum / max(self.n, 1)

    def nll(self):
        return self.nll_sum / max(self.n, 1)

    def risk_coverage(self):
        """Coverage and selective risk when keeping the most confident pixels."""
        n = self.rc_n[::-1].cumsum()
        c = self.rc_c[::-1].cumsum()
        cov = n / n[-1]
        risk = 1 - c / np.maximum(n, 1)
        keep = self.rc_n[::-1] > 0
        return cov[keep], risk[keep]

    def aurc(self):
        cov, risk = self.risk_coverage()
        cov = np.concatenate([[0.0], cov])
        risk = np.concatenate([[risk[0]], risk])
        return float(np.trapz(risk, cov))

    def state(self):
        return dict(ece_n=self.ece_n, ece_c=self.ece_c, ece_s=self.ece_s, rc_n=self.rc_n, rc_c=self.rc_c,
                    brier_sum=self.brier_sum, nll_sum=self.nll_sum, n=self.n)


def binary_auroc_from_hist(rc_n, rc_c):
    """AUROC of a confidence score for separating correct vs wrong pixels."""
    pos = rc_c  # correct
    neg = rc_n - rc_c
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float('nan')
    cneg = np.concatenate([[0], np.cumsum(neg)[:-1]])
    return float(np.sum(pos * (cneg + 0.5 * neg)) / (P * N))
