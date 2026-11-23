"""
ensemble_eval.py   (run from the Bluewatch folder:  py -3.12 ensemble_eval.py)

No training. It takes the U-Nets you already trained and measures, on the TEST crops:
  1. each model alone
  2. all models averaged together (ensemble)
  3. the ensemble + test-time flips (each crop is read 4 ways and the answers are averaged)
It prints pixel accuracy, mIoU, oil IoU/recall/precision and oil-vs-everything-else accuracy.
Nothing is tuned on the test set. It only reads files.
"""
import glob
import os
import sys
import warnings

import numpy as np
import rasterio
import torch

warnings.filterwarnings("ignore")
sys.argv = sys.argv[:1]
import unet_train as ut  # noqa: E402

OIL = ut.OIL - 1
files = [f for f in ["bluewatch_unet.pt"] + sorted(glob.glob("bluewatch_unet_v2_seed*.pt")) if os.path.exists(f)]
if not files:
    sys.exit("No model files found in this folder.")
print("models:", files, "| device:", ut.DEVICE, flush=True)

models = []
for f in files:
    m = ut.UNet().to(ut.DEVICE)
    m.load_state_dict(torch.load(f, map_location=ut.DEVICE))
    m.eval()
    models.append(m)

names = open(f"{ut.ROOT}/splits/test_X.txt").read().split()
variants = [f"alone: {f}" for f in files] + ["ensemble", "ensemble + flips"]
cms = [np.zeros((15, 15), dtype=np.int64) for _ in variants]


@torch.no_grad()
def probs(m, x):
    return torch.softmax(m(x).float(), 1)


@torch.no_grad()
def probs_flips(m, x):
    p = probs(m, x)
    p = p + probs(m, x.flip(-1)).flip(-1)
    p = p + probs(m, x.flip(-2)).flip(-2)
    p = p + probs(m, x.flip(-1, -2)).flip(-1, -2)
    return p / 4


for k, name in enumerate(names, 1):
    scene, crop = name.rsplit("_", 1)
    ip = f"{ut.ROOT}/{scene}/{scene}_L2R_rhorc_{crop}.tif"
    lp = f"{ut.ROOT}/{scene}/{scene}_L2R_cl_{crop}.tif"
    if not (os.path.exists(ip) and os.path.exists(lp)):
        continue
    with rasterio.open(ip) as s:
        x = s.read().astype(np.float32)
    with rasterio.open(lp) as s:
        y = s.read(1)
    if x.shape != (11, 240, 240) or y.shape != (240, 240):
        continue
    x = np.where(np.isfinite(x), x, ut.MEAN[:, None, None])
    x = np.clip((x - ut.MEAN[:, None, None]) / ut.STD[:, None, None], -10, 40).astype(np.float32)
    xt = torch.from_numpy(x)[None].to(ut.DEVICE)
    m_ = (y >= 1) & (y <= 15)
    if not m_.any():
        continue
    t = y[m_].astype(int) - 1
    single = [probs(m, xt) for m in models]
    outs = single + [sum(single) / len(single), sum(probs_flips(m, xt) for m in models) / len(models)]
    for cm, p in zip(cms, outs):
        pred = p.argmax(1)[0].cpu().numpy()[m_]
        np.add.at(cm, (t, pred), 1)
    if k % 150 == 0:
        print(f"  {k}/{len(names)} crops", flush=True)


def summary(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    total = cm.sum()
    oil_rest = (tp[OIL] + (total - cm[OIL].sum() - fp[OIL])) / total
    return (100 * tp.sum() / total, iou[cm.sum(1) > 0].mean(), iou[OIL],
            tp[OIL] / max(tp[OIL] + fn[OIL], 1), tp[OIL] / max(tp[OIL] + fp[OIL], 1), 100 * oil_rest)


print(f"\n=== TEST SET ({int(cms[0].sum())} labelled pixels, scenes never used for training) ===")
print(f"{'model':45s} {'pix acc':>8s} {'mIoU':>6s} {'oilIoU':>7s} {'recall':>7s} {'prec':>6s} {'oil/not':>8s}")
for v, cm in zip(variants, cms):
    a, mi, oi, rc, pr, orr = summary(cm)
    print(f"{v:45s} {a:7.1f}% {mi:6.3f} {oi:7.3f} {rc:7.3f} {pr:6.3f} {orr:7.1f}%")
