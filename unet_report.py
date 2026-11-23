"""
unet_report.py
Run from your Bluewatch folder, with the Python that has PyTorch:

    py -3.12 unet_report.py

1. Scores the saved U-Net (bluewatch_unet.pt) on every TEST crop and prints the plain numbers
   a teacher asks for: pixel accuracy, mIoU, oil IoU and oil-versus-everything-else accuracy.
2. Lists test crops that work well as a live demo, with the exact file names to upload.

It only reads files. It does not change any model.
It needs unet_train.py in the same folder (it reuses the model from there).
"""
import os
import sys
import warnings

import numpy as np
import rasterio
import torch

warnings.filterwarnings("ignore")
sys.argv = sys.argv[:1]            # make sure unet_train does not think this is a "quick" run
import unet_train as ut            # noqa: E402  (only reads the model class and the normalisation)

ROOT = ut.ROOT
OIL = ut.OIL - 1                   # class index of Oil Spill, 0..14

model = ut.UNet().to(ut.DEVICE)
model.load_state_dict(torch.load("bluewatch_unet.pt", map_location=ut.DEVICE))
model.eval()
print("device:", ut.DEVICE, flush=True)

names = open(f"{ROOT}/splits/test_X.txt").read().split()
cm = np.zeros((15, 15), dtype=np.int64)          # rows = real class, columns = predicted class
rows = []

for k, name in enumerate(names, 1):
    scene, crop = name.rsplit("_", 1)
    ip = f"{ROOT}/{scene}/{scene}_L2R_rhorc_{crop}.tif"
    lp = f"{ROOT}/{scene}/{scene}_L2R_cl_{crop}.tif"
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
    with torch.no_grad():
        pred = model(torch.from_numpy(x)[None].to(ut.DEVICE)).argmax(1)[0].cpu().numpy()

    m = (y >= 1) & (y <= 15)                      # only pixels that experts labelled
    t = y[m].astype(int) - 1
    p = pred[m]
    np.add.at(cm, (t, p), 1)
    oil_t, oil_p = t == OIL, p == OIL
    rows.append(dict(name=name, scene=scene, crop=crop, n=int(m.sum()), correct=int((t == p).sum()),
                     oil=int(oil_t.sum()), other=int((~oil_t).sum()),
                     tp=int((oil_t & oil_p).sum()), fp=int((~oil_t & oil_p).sum()), fn=int((oil_t & ~oil_p).sum())))
    if k % 100 == 0:
        print(f"  {k}/{len(names)} crops done", flush=True)

tp = np.diag(cm).astype(float)
fp = cm.sum(0) - tp
fn = cm.sum(1) - tp
iou = tp / np.maximum(tp + fp + fn, 1)
present = cm.sum(1) > 0
total = int(cm.sum())
accuracy = tp.sum() / total
not_oil_right = (total - cm[OIL].sum()) - fp[OIL]            # real non-oil pixels that were not flagged as oil
oil_vs_rest = (tp[OIL] + not_oil_right) / total

print("\n=== U-NET ON THE TEST SET ===")
print(f"labelled pixels: {total}")
print(f"pixel accuracy, all 15 classes: {100 * accuracy:.1f}%   (share of labelled pixels given the right class)")
print(f"mIoU: {iou[present].mean():.3f}   (earlier run: 0.573)")
print(f"oil IoU: {iou[OIL]:.3f}   (earlier run: 0.386) | oil recall: {tp[OIL] / max(tp[OIL] + fn[OIL], 1):.3f}"
      f" | oil precision: {tp[OIL] / max(tp[OIL] + fp[OIL], 1):.3f}")
print(f"oil-versus-everything-else accuracy: {100 * oil_vs_rest:.1f}%   (only asks: is this pixel oil or not oil?)")


def oil_iou(r):
    d = r["tp"] + r["fp"] + r["fn"]
    return r["tp"] / d if d else 0.0


oilrows = sorted((r for r in rows if r["oil"] >= 100), key=lambda r: (oil_iou(r), r["oil"]), reverse=True)
print("\n=== CROPS WITH REAL OIL (best first) ===")
print(f"  crops with at least 100 real oil pixels: {len(oilrows)}")
for r in oilrows[:10]:
    print(f"  {r['name']:14s} oil pixels {r['oil']:5d} | found {r['tp']:5d} | false flags {r['fp']:5d} | "
          f"oil IoU {oil_iou(r):.2f} | crop accuracy {100 * r['correct'] / r['n']:.0f}%")

clean = sorted((r for r in rows if r["oil"] == 0 and r["n"] >= 800 and r["fp"] == 0),
               key=lambda r: r["n"], reverse=True)
print("\n=== CROPS WITH NO OIL, WHERE THE U-NET RAISES NO FALSE ALARM ===")
for r in clean[:5]:
    print(f"  {r['name']:14s} labelled pixels {r['n']:5d} | crop accuracy {100 * r['correct'] / r['n']:.0f}% | oil flags 0")

if oilrows:
    r = oilrows[0]
    print(f"\nTo upload the best one, choose these two files in  {ROOT}/{r['scene']}/ :")
    print(f"  {r['scene']}_L2R_rhorc_{r['crop']}.tif   and   {r['scene']}_L2R_cl_{r['crop']}.tif")
print("\nThese crops are examples picked after the fact. Say so, and give the overall numbers above as the real score.")