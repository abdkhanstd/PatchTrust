"""Parameter count, FLOPs (fvcore-free MAC count via hooks), latency (GPU fp32/fp16,
CPU fp32, batch 1, 256x352) and peak inference memory for the base and PatchTrust."""
import argparse
import json
import time
import torch
import torch.nn as nn

from .models import Base, PatchTrust, count_params


def macs(model, x):
    tot = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            tot[0] += o.numel() * k
        elif isinstance(m, nn.Linear):
            tot[0] += o.numel() * m.in_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return tot[0]


def latency(model, x, n=200, warm=30, half=False):
    dev = x.device
    import contextlib
    ctx = torch.autocast('cuda', dtype=torch.float16) if (half and dev.type == 'cuda') else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        for _ in range(warm):
            model(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            model(x)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    ts.sort()
    return 1000 * ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    res = {}
    base = Base(11, pretrained_backbone=False).eval()
    pt = PatchTrust(Base(11, pretrained_backbone=False), gate='learned').eval()
    for name, m in [('base', base), ('patchtrust', pt)]:
        r = {'params': count_params(m), 'params_heads': count_params(m.heads) if name == 'patchtrust' else 0}
        x = torch.randn(1, 3, 256, 352)
        r['gmacs'] = macs(m, x) / 1e9
        torch.set_num_threads(4)
        r['cpu4_ms'] = latency(m, x, n=50, warm=5)
        if torch.cuda.is_available():
            mc, xc = m.cuda(), x.cuda()
            r['gpu_fp32_ms'] = latency(mc, xc)
            r['gpu_fp16_ms'] = latency(mc, xc, half=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                mc(xc)
            r['gpu_peak_infer_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
            m.cpu()
        res[name] = r
        print(name, r, flush=True)
    res['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
