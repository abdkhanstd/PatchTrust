"""LR-ASPP MobileNetV3 base model and the PatchTrust heads.

PatchTrust adds two lightweight heads on top of a (frozen) base segmenter:
  * reliability head  -> r(x)  = P(base prediction correct)       (gate)
                         q(x)  = P(final prediction correct)      (output confidence)
  * correction head   -> dL(x) = residual class-logit correction
and composes  L_final = L_base + g(x) * dL(x)  with g = 1 - r for the learned gate.
Other gates (ungated, entropy, unsupervised learned, privileged oracle) are
provided for ablations.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x_uint8_nhwc):
    """uint8 NHWC tensor -> float NCHW, ImageNet-normalised."""
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Base(nn.Module):
    """arch='large': torchvision LR-ASPP MobileNetV3-Large.
    arch='small': the same LR-ASPP construction on a MobileNetV3-Small backbone."""

    def __init__(self, num_classes=11, pretrained_backbone=True, arch='large'):
        super().__init__()
        self.arch = arch
        if arch == 'large':
            from torchvision.models import MobileNet_V3_Large_Weights
            kw = dict(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
            m = lraspp_mobilenet_v3_large(num_classes=num_classes, **kw)
        else:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            from torchvision.models.segmentation.lraspp import _lraspp_mobilenetv3
            bb = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None, dilated=True)
            m = _lraspp_mobilenetv3(bb, num_classes)
        self.backbone = m.backbone
        self.classifier = m.classifier
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, 64, 64))
        self.low_ch, self.high_ch = f['low'].shape[1], f['high'].shape[1]

    def forward_all(self, x):
        f = self.backbone(x)
        return f['low'], f['high'], self.classifier(f)  # logits at stride 8

    def forward(self, x):
        _, _, lo = self.forward_all(x)
        return F.interpolate(lo, size=x.shape[-2:], mode='bilinear', align_corners=False)


def dsconv(cin, cout, dil=1):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=dil, dilation=dil, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


_SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])


def grad_magnitude(x):
    """Sobel gradient magnitude of the luminance of an ImageNet-normalised image."""
    rgb = x * STD.to(x.device) + MEAN.to(x.device)
    g = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    kx = _SOBEL.to(x.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def cues(x, base_lo):
    """Cheap per-pixel cues at stride 8: normalised entropy, max-prob, mean and
    max local gradient magnitude."""
    p = base_lo.float().softmax(1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1, keepdim=True) / math.log(p.shape[1])
    mp = p.max(1, keepdim=True)[0]
    gm = grad_magnitude(x)
    s = x.shape[-1] // base_lo.shape[-1]
    gavg = F.adaptive_avg_pool2d(gm, base_lo.shape[-2:])
    gmax = F.adaptive_max_pool2d(gm, base_lo.shape[-2:])
    return torch.cat([ent, mp, gavg, gmax / 4.0], 1), p


class PatchTrustHeads(nn.Module):
    def __init__(self, num_classes=11, low_ch=40, high_ch=960, mid=64, corr_ch=96,
                 use_cues=True, selector=False, corr_pb=True, rel_feats=True):
        super().__init__()
        self.use_cues = use_cues
        self.corr_pb = corr_pb      # corrector sees the base softmax
        self.rel_feats = rel_feats  # reliability head sees encoder features (else cues only)
        self.selector = selector
        nc = 4 if use_cues else 0
        self.high_proj = nn.Sequential(nn.Conv2d(high_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        rin = (low_ch + mid if rel_feats else 0) + (4 if (use_cues or not rel_feats) else 0)
        self.rel = nn.Sequential(dsconv(rin, 64), dsconv(64, 64, dil=2), nn.Conv2d(64, 3, 1))
        rin = low_ch + mid + nc
        cin = low_ch + mid + (num_classes if corr_pb else 0) + nc
        self.corr = nn.Sequential(dsconv(cin, corr_ch), dsconv(corr_ch, corr_ch, dil=2),
                                  dsconv(corr_ch, corr_ch, dil=4), nn.Conv2d(corr_ch, num_classes, 1))
        nn.init.zeros_(self.corr[-1].weight)
        nn.init.zeros_(self.corr[-1].bias)
        nn.init.constant_(self.rel[-1].bias, 2.0)  # start near "reliable"
        if selector:  # arbitration head: sees both the base and the corrected prediction
            self.sel = nn.Sequential(dsconv(rin + 2 * num_classes, 64), dsconv(64, 64, dil=2), nn.Conv2d(64, 1, 1))

    def forward(self, x, low, high, base_lo):
        c, p = cues(x, base_lo)
        h = F.interpolate(self.high_proj(high), size=low.shape[-2:], mode='bilinear', align_corners=False)
        feats = [low, h] + ([c] if self.use_cues else [])
        rel_in = (feats[:2] if self.rel_feats else []) + ([c.to(low.dtype)] if (self.use_cues or not self.rel_feats) else [])
        rel = self.rel(torch.cat(rel_in, 1))  # ch0: base-correct logit, ch1: final-correct logit, ch2: free gate logit
        dl = self.corr(torch.cat([low, h] + ([p.to(low.dtype)] if self.corr_pb else []) + ([c] if self.use_cues else []), 1))
        sel = None
        if self.selector:
            pc = (base_lo.float() + dl.float()).softmax(1).detach()
            sel = self.sel(torch.cat(feats + [p.to(low.dtype), pc.to(low.dtype)], 1))
        return rel, dl, c, sel


class PatchTrust(nn.Module):
    """gate in {'none','ungated','entropy','learned','learned_unsup','oracle'}"""

    def __init__(self, base: Base, gate='learned', num_classes=11, gate_grad=False, use_cues=True, gate_thresh=None,
                 corr_pb=True, rel_feats=True):
        super().__init__()
        self.gate_thresh = gate_thresh  # optional hard threshold on r at inference
        self.base = base
        self.gate = gate
        self.gate_grad = gate_grad
        self.heads = PatchTrustHeads(num_classes, low_ch=base.low_ch, high_ch=base.high_ch, use_cues=use_cues,
                                     selector=(gate == 'select'), corr_pb=corr_pb, rel_feats=rel_feats) if gate != 'none' else None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # base is frozen, BN statistics fixed
        return self

    def forward(self, x, oracle_correct=None, y=None):
        size = x.shape[-2:]
        with torch.no_grad():
            low, high, base_lo = self.base.forward_all(x)
        up = lambda t: F.interpolate(t, size=size, mode='bilinear', align_corners=False)
        Lb = up(base_lo).float()
        out = {'base': Lb}
        if self.heads is None:
            out['final'] = Lb
            return out
        rel, dl, c, sel = self.heads(x, low, high, base_lo)
        rel, dl = up(rel).float(), up(dl).float()
        r = torch.sigmoid(rel[:, 0:1])
        out.update(r_logit=rel[:, 0:1], q_logit=rel[:, 1:2], r=r, q=torch.sigmoid(rel[:, 1:2]), dl=dl)
        if self.gate == 'ungated':
            g = torch.ones_like(r)
        elif self.gate == 'entropy':
            g = up(c[:, 0:1]).float()
        elif self.gate == 'learned' or (self.gate == 'learned_tf' and (not self.training or y is None)):
            g = 1 - (r if self.gate_grad else r.detach())
            if self.gate_thresh is not None and not self.training:
                g = (r < self.gate_thresh).float()
        elif self.gate == 'learned_tf':  # training: corrector is taught with the true error mask (teacher forcing)
            g = ((Lb.argmax(1) != y) & (y != 255)).float().unsqueeze(1)
        elif self.gate == 'select':
            s_logit = up(sel).float()
            out['s_logit'] = s_logit
            if self.training and y is not None:  # corrector trained on the true error mask
                g = ((Lb.argmax(1) != y) & (y != 255)).float().unsqueeze(1)
                out['final_sel'] = Lb + torch.sigmoid(s_logit) * dl.detach()
            else:
                g = torch.sigmoid(s_logit)
        elif self.gate == 'learned_unsup':
            g = torch.sigmoid(rel[:, 2:3])
        elif self.gate == 'oracle':  # PRIVILEGED: uses ground truth, analysis only
            g = 1 - oracle_correct.float().unsqueeze(1)
        else:
            raise ValueError(self.gate)
        out['g'] = g
        out['final'] = Lb + g * dl
        return out


def count_params(m):
    return sum(p.numel() for p in m.parameters())
