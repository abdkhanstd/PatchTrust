"""Camera corruptions on uint8 HWC RGB images, three severities.

SEEN corruptions are used for training augmentation and evaluation.
HELDOUT corruptions are never used in training and test generalisation.
All functions are deterministic given the numpy Generator.
"""
import numpy as np
import cv2

SEEN = ['gaussian_noise', 'gaussian_blur', 'low_light', 'jpeg', 'occlusion', 'local_blur']
HELDOUT = ['motion_blur', 'contrast', 'fog', 'pixelate']
ALL = SEEN + HELDOUT


def _f(x):
    return x.astype(np.float32)


def _u8(x):
    return np.clip(x, 0, 255).round().astype(np.uint8)


def gaussian_noise(im, s, rng):
    sigma = [0.06, 0.12, 0.20][s - 1] * 255
    return _u8(_f(im) + rng.normal(0, sigma, im.shape).astype(np.float32))


def gaussian_blur(im, s, rng):
    sig = [1.0, 2.0, 3.5][s - 1]
    return cv2.GaussianBlur(im, (0, 0), sig)


def low_light(im, s, rng):
    f = [0.5, 0.3, 0.15][s - 1]
    return _u8(_f(im) * f)


def jpeg(im, s, rng):
    q = [25, 12, 6][s - 1]
    ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(im[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def _rect(rng, H, W, frac):
    ar = rng.uniform(0.5, 2.0)
    area = frac * H * W
    h = int(min(H, max(4, np.sqrt(area / ar))))
    w = int(min(W, max(4, area / max(h, 1))))
    y = int(rng.integers(0, H - h + 1))
    x = int(rng.integers(0, W - w + 1))
    return y, x, h, w


def occlusion(im, s, rng):
    """Opaque rectangles (random flat colour or noise), total area 5/10/20 %."""
    frac = [0.05, 0.10, 0.20][s - 1]
    out = im.copy()
    H, W = im.shape[:2]
    n = int(rng.integers(1, 4))
    for _ in range(n):
        y, x, h, w = _rect(rng, H, W, frac / n)
        if rng.random() < 0.5:
            out[y:y + h, x:x + w] = rng.integers(0, 256, 3).astype(np.uint8)
        else:
            out[y:y + h, x:x + w] = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    return out


def local_blur(im, s, rng):
    """One rectangle (15/25/35 % area) blurred strongly, e.g. a smudge on the lens."""
    frac = [0.15, 0.25, 0.35][s - 1]
    sig = [3.0, 5.0, 8.0][s - 1]
    H, W = im.shape[:2]
    y, x, h, w = _rect(rng, H, W, frac)
    out = im.copy()
    out[y:y + h, x:x + w] = cv2.GaussianBlur(im, (0, 0), sig)[y:y + h, x:x + w]
    return out


def motion_blur(im, s, rng):
    L = [7, 13, 19][s - 1]
    ang = rng.uniform(0, 180)
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= k.sum()
    return cv2.filter2D(im, -1, k)


def contrast(im, s, rng):
    c = [0.5, 0.35, 0.2][s - 1]
    x = _f(im)
    m = x.mean(axis=(0, 1), keepdims=True)
    return _u8((x - m) * c + m)


def fog(im, s, rng):
    """Spatially varying haze: I*t + A*(1-t), t from a smooth random field."""
    strength = [0.4, 0.6, 0.75][s - 1]
    H, W = im.shape[:2]
    field = rng.uniform(0, 1, (max(H // 32, 2), max(W // 32, 2))).astype(np.float32)
    field = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), 8)
    field = 0.6 + 0.4 * (field - field.min()) / (field.max() - field.min() + 1e-6)
    t = 1 - strength * field
    A = 0.85 * 255
    return _u8(_f(im) * t[..., None] + A * (1 - t[..., None]))


def pixelate(im, s, rng):
    f = [0.5, 0.33, 0.2][s - 1]
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


FN = {k: globals()[k] for k in ALL}


def apply_corruption(im, name, severity, rng):
    assert 1 <= severity <= 3
    out = FN[name](np.ascontiguousarray(im), severity, rng)
    assert out.shape == im.shape and out.dtype == np.uint8
    return out
