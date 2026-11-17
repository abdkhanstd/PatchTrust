import os
import numpy as np
import torch
import pytest

from patchtrust.corruptions import ALL, apply_corruption
from patchtrust.models import Base, PatchTrust, normalize, count_params
from patchtrust.metrics import per_image_iu, miou_from, boundary_counts, bf1_from, CalibAcc
from patchtrust.data import CamVid, FixedCorrupted

DATA = os.environ.get('CAMVID', os.path.join(os.path.dirname(__file__), '..', 'data_dl', 'segnet', 'CamVid'))


@pytest.mark.parametrize('name', ALL)
@pytest.mark.parametrize('sev', [1, 2, 3])
def test_corruptions_deterministic(name, sev):
    im = np.random.RandomState(0).randint(0, 256, (64, 88, 3)).astype(np.uint8)
    a = apply_corruption(im, name, sev, np.random.default_rng([1, 2, 3]))
    b = apply_corruption(im, name, sev, np.random.default_rng([1, 2, 3]))
    assert a.shape == im.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_model_forward_backward_and_identity_init():
    base = Base(11, pretrained_backbone=False).eval()
    m = PatchTrust(base, gate='learned')
    x = normalize(torch.randint(0, 256, (2, 64, 96, 3), dtype=torch.uint8))
    out = m.train()(x)
    assert out['final'].shape == (2, 11, 64, 96)
    # zero-initialised correction: final == base at start
    assert torch.allclose(out['final'], out['base'])
    y = torch.randint(0, 11, (2, 64, 96))
    loss = torch.nn.functional.cross_entropy(out['final'], y) + out['r_logit'].mean()
    loss.backward()
    assert all(p.grad is None for p in m.base.parameters())
    assert m.heads.corr[-1].weight.grad is not None
    assert count_params(m.heads) < 0.2 * count_params(base)


def test_metrics_toy():
    lab = torch.tensor([[[0, 0, 1, 255]]])
    pred = torch.tensor([[[0, 1, 1, 0]]])
    I, U, c, v = per_image_iu(pred, lab, 2)
    assert I.tolist() == [[1, 1]] and U.tolist() == [[2, 2]]
    assert c.item() == 2 and v.item() == 3
    m, _ = miou_from(I.sum(0), U.sum(0))
    assert abs(m - 0.5) < 1e-9
    a = CalibAcc()
    a.add(torch.tensor([[[1.0, 1.0, 0.0, 0.5]]]), torch.tensor([[[True, True, False, True]]]), lab != 255)
    assert a.ece() < 1e-6


def test_boundary_perfect():
    lab = torch.zeros(1, 20, 20, dtype=torch.long)
    lab[:, :, 10:] = 1
    assert abs(bf1_from(boundary_counts(lab, lab)) - 1.0) < 1e-9


@pytest.mark.skipif(not os.path.isdir(DATA), reason='CamVid not present')
def test_camvid_split_and_labels():
    for split, n in [('train', 367), ('val', 101), ('test', 233)]:
        ds = CamVid(DATA, split)
        assert len(ds) == n
    ds = CamVid(DATA, 'train', train=True, corrupt_p=1.0, subset=8)
    xc, xx, y, cid = ds[0]
    assert xc.shape == (256, 352, 3) and y.shape == (256, 352)
    assert set(np.unique(y.numpy())).issubset(set(range(11)) | {255})
    te = CamVid(DATA, 'test')
    names_tr = set(CamVid(DATA, 'train').names)
    assert not names_tr & set(te.names)
    f = FixedCorrupted(te, 'jpeg', 2)
    assert torch.equal(f[3][0], f[3][0])
