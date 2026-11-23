"""
analysis_report.py   (run from the Bluewatch folder:  py -3.12 analysis_report.py)

No training. Uses every U-Net it finds (bluewatch_unet.pt, v2 seeds, v3 seeds) as an ensemble and measures, on the TEST crops:
  1. a baseline table (what a lazy "everything is water" guess scores, versus the models)
  2. where the real oil goes (which class the model calls it) and what gets wrongly called oil
  3. a precision-recall curve for oil (every slider setting, not just 50%)
  4. the 4 crops where the model did worst on oil (failure gallery)
Files go to the folder  analysis/  . Optional quick check on fewer crops:  py -3.12 analysis_report.py 60
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
sys.argv = sys.argv[:1]
import unet_train as ut  # noqa: E402

OIL = ut.OIL - 1
NAMES = ut.CLASS_NAMES
os.makedirs("analysis", exist_ok=True)

files = [f for f in ["bluewatch_unet.pt"] + sorted(glob.glob("bluewatch_unet_v2_seed*.pt")) + sorted(glob.glob("bluewatch_unet_v3_seed*.pt")) if os.path.exists(f)]
if not files:
    sys.exit("No model files found in this folder.")
print("models:", files, "| device:", ut.DEVICE, flush=True)
models = []
for f in files:
    sd = torch.load(f, map_location=ut.DEVICE)
    m = ut.UNet(c=sd["e1.net.0.weight"].shape[0]).to(ut.DEVICE)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)

names = open(f"{ut.ROOT}/splits/test_X.txt").read().split()
if LIMIT:
    names = names[:LIMIT]
cm = np.zeros((15, 15), dtype=np.int64)
BINS = 1000
pos_hist = np.zeros(BINS, dtype=np.int64)     # oil probability of real-oil pixels
neg_hist = np.zeros(BINS, dtype=np.int64)     # oil probability of everything else
rows = []

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
    xn = np.clip((x - ut.MEAN[:, None, None]) / ut.STD[:, None, None], -10, 40).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(xn)[None].to(ut.DEVICE)
        p = sum(torch.softmax(m(xt).float(), 1) for m in models)[0] / len(models)
    p = p.cpu().numpy()
    pred = p.argmax(0)
    lab = (y >= 1) & (y <= 15)
    if not lab.any():
        continue
    t = y[lab].astype(int) - 1
    np.add.at(cm, (t, pred[lab]), 1)
    po = p[OIL][lab]
    b = np.minimum((po * BINS).astype(int), BINS - 1)
    pos_hist += np.bincount(b[t == OIL], minlength=BINS)
    neg_hist += np.bincount(b[t != OIL], minlength=BINS)
    oil_t, oil_p = t == OIL, pred[lab] == OIL
    rows.append(dict(name=name, scene=scene, crop=crop, oil=int(oil_t.sum()), tp=int((oil_t & oil_p).sum()),
                     fp=int((~oil_t & oil_p).sum()), fn=int((oil_t & ~oil_p).sum()), n=int(lab.sum())))
    if k % 150 == 0:
        print(f"  {k}/{len(names)} crops", flush=True)

total = int(cm.sum())
tp = np.diag(cm).astype(float)
fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
iou = tp / np.maximum(tp + fp + fn, 1)
acc = tp.sum() / total
share = cm.sum(1) / total
lazy = share[6]                                   # "everything is Marine Water"
print(f"\nlabelled test pixels: {total}")

# ---------------- 1. baseline table
with open("analysis/baseline_table.txt", "w") as f:
    f.write("What each guess scores on the test set (pixel accuracy over labelled pixels)\n")
    f.write(f"  Guess 'everything is Marine Water' : {100 * lazy:.1f}%\n")
    f.write(f"  Random Forest (earlier run)         : see report.json\n")
    f.write(f"  This ensemble ({len(models)} U-Nets)           : {100 * acc:.1f}%   mIoU {iou[cm.sum(1) > 0].mean():.3f}\n")
print(open("analysis/baseline_table.txt").read())

# ---------------- 2. oil confusion
oil_row = cm[OIL] / max(cm[OIL].sum(), 1)
oil_col = cm[:, OIL] / max(cm[:, OIL].sum(), 1)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
order = np.argsort(-oil_row)[:7]
ax[0].barh([NAMES[i] for i in order][::-1], (100 * oil_row[order])[::-1], color=["#e11d1d" if i == OIL else "#6b7bb3" for i in order][::-1])
ax[0].set_title("Real oil pixels: what the model called them (%)")
order2 = np.argsort(-oil_col)[:7]
ax[1].barh([NAMES[i] for i in order2][::-1], (100 * oil_col[order2])[::-1], color=["#e11d1d" if i == OIL else "#c98a2b" for i in order2][::-1])
ax[1].set_title("Pixels the model called oil: what they really were (%)")
for a in ax:
    a.set_xlabel("%")
fig.tight_layout()
fig.savefig("analysis/oil_confusion.png", dpi=120)
plt.close(fig)
print("Real oil was called:")
for i in np.argsort(-oil_row)[:5]:
    print(f"  {NAMES[i]:28s} {100 * oil_row[i]:5.1f}%")
print("Pixels called oil were really:")
for i in np.argsort(-oil_col)[:5]:
    print(f"  {NAMES[i]:28s} {100 * oil_col[i]:5.1f}%")

# ---------------- 3. precision-recall for oil
cp, cn = np.cumsum(pos_hist[::-1])[::-1], np.cumsum(neg_hist[::-1])[::-1]      # pixels at or above each threshold
recall = cp / max(pos_hist.sum(), 1)
precision = np.where(cp + cn > 0, cp / np.maximum(cp + cn, 1), 1.0)
thr = np.arange(BINS) / BINS
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(recall, precision, color="#e11d1d", lw=2)
for t_ in (0.3, 0.5, 0.7, 0.9):
    i = int(t_ * BINS)
    ax.scatter([recall[i]], [precision[i]], color="#111", zorder=3)
    ax.annotate(f"{int(t_ * 100)}%", (recall[i], precision[i]), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xlabel("Oil caught (recall)")
ax.set_ylabel("Oil flags that are right (precision)")
ax.set_title("Oil: catching more vs. fewer false alarms\n(labels = the slider setting)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("analysis/pr_curve.png", dpi=120)
plt.close(fig)
points = {}
print("\nSlider setting -> oil caught / flags right:")
for t_ in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    i = int(t_ * BINS)
    points[str(t_)] = [round(float(recall[i]), 3), round(float(precision[i]), 3)]
    print(f"  {int(t_ * 100):3d}%  caught {100 * recall[i]:5.1f}%   right {100 * precision[i]:5.1f}%")

# ---------------- 4. failure gallery (worst 4 crops that have real oil)
bad = sorted((r for r in rows if r["oil"] >= 100), key=lambda r: (r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1), -r["oil"]))[:4]
base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:15]
base[OIL] = (0.9, 0.05, 0.05, 1)
base[6] = (0.12, 0.2, 0.45, 1)
cmap = ListedColormap(np.vstack([[0, 0, 0, 1], base]))
gal = []
for r in bad:
    scene, crop = r["scene"], r["crop"]
    with rasterio.open(f"{ut.ROOT}/{scene}/{scene}_L2R_rhorc_{crop}.tif") as s:
        x = s.read().astype(np.float32)
    with rasterio.open(f"{ut.ROOT}/{scene}/{scene}_L2R_cl_{crop}.tif") as s:
        y = s.read(1).astype(int)
    xx = np.where(np.isfinite(x), x, ut.MEAN[:, None, None])
    xn = np.clip((xx - ut.MEAN[:, None, None]) / ut.STD[:, None, None], -10, 40).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(xn)[None].to(ut.DEVICE)
        pr = (sum(torch.softmax(m(xt).float(), 1) for m in models)[0] / len(models)).argmax(0).cpu().numpy() + 1
    truth = np.where((y >= 1) & (y <= 15), y, 0)
    shown = np.where(truth > 0, pr, 0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    ax[0].imshow(np.clip(xx[[3, 2, 1]].transpose(1, 2, 0) / 0.15, 0, 1)); ax[0].set_title("Satellite")
    ax[1].imshow(cmap(shown), interpolation="nearest"); ax[1].set_title("Model")
    ax[2].imshow(cmap(truth), interpolation="nearest"); ax[2].set_title("Real labels")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{scene}, crop {crop}: real oil {r['oil']} px, found {r['tp']}, false flags {r['fp']}")
    fig.tight_layout()
    fn_ = f"analysis/failure_{scene}_crop{crop}.png"
    fig.savefig(fn_, dpi=100)
    plt.close(fig)
    gal.append(dict(file=fn_, scene=scene, crop=crop, real_oil=r["oil"], found=r["tp"], false_flags=r["fp"]))
    print("saved", fn_)

json.dump(dict(models=files, labelled_pixels=total, pixel_accuracy=round(float(acc), 4), miou=round(float(iou[cm.sum(1) > 0].mean()), 4),
               lazy_all_water=round(float(lazy), 4),
               oil_called=[{"class": NAMES[i], "pct": round(float(100 * oil_row[i]), 1)} for i in np.argsort(-oil_row)[:6]],
               called_oil_really=[{"class": NAMES[i], "pct": round(float(100 * oil_col[i]), 1)} for i in np.argsort(-oil_col)[:6]],
               slider_points=points, failures=gal), open("analysis/analysis.json", "w"), indent=1)
print("\nDone. Open the analysis folder.")
