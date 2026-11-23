"""
make_demo_images.py
Run from the Bluewatch folder:   py -3.12 make_demo_images.py

For every crop in demo_files/ (the two files rhorc + cl), it runs the saved U-Net (bluewatch_unet.pt)
and saves one picture per crop in demo_results/ with three panels:
    satellite (true colour) | U-Net prediction | real labels from the experts
Oil is drawn in red. Nothing is edited or faked: the middle panel is the model's real output.
"""
import glob
import os
import re
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
sys.argv = sys.argv[:1]
import unet_train as ut  # noqa: E402

FOLDER, OUT = "demo_files", "demo_results"
os.makedirs(OUT, exist_ok=True)

model = ut.UNet().to(ut.DEVICE)
model.load_state_dict(torch.load("bluewatch_unet.pt", map_location=ut.DEVICE))
model.eval()

base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:15]
colors = base.copy()
colors[ut.OIL - 1] = (0.9, 0.05, 0.05, 1)                 # oil = red
colors[6] = (0.12, 0.2, 0.45, 1)                          # marine water = dark blue
cmap = ListedColormap(np.vstack([[0, 0, 0, 1], colors]))  # index 0 = unlabelled (black)


def paint(mask):                                          # mask: 0 = unlabelled, 1..15 = class
    return cmap(mask)


files = sorted(glob.glob(f"{FOLDER}/*_L2R_rhorc_*.tif"))
if not files:
    sys.exit(f"No *_L2R_rhorc_*.tif files found in {FOLDER}/")

for ip in files:
    m = re.search(r"(Scene_\d+)_L2R_rhorc_(\d+)\.tif$", os.path.basename(ip))
    if not m:
        print("skipped (file name not recognised):", ip, flush=True)
        continue
    scene, crop = m.group(1), m.group(2)
    lp = f"{FOLDER}/{scene}_L2R_cl_{crop}.tif"
    with rasterio.open(ip) as s:
        x = s.read().astype(np.float32)
    x = np.where(np.isfinite(x), x, ut.MEAN[:, None, None])
    xn = np.clip((x - ut.MEAN[:, None, None]) / ut.STD[:, None, None], -10, 40).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(xn)[None].to(ut.DEVICE)).argmax(1)[0].cpu().numpy() + 1
    rgb = np.clip(x[[3, 2, 1]].transpose(1, 2, 0) / 0.15, 0, 1)   # bands b4, b3, b2, brightened

    have_truth = os.path.exists(lp)
    if have_truth:
        with rasterio.open(lp) as s:
            truth = s.read(1).astype(int)
        truth = np.where((truth >= 1) & (truth <= 15), truth, 0)
        lab = truth > 0
        oil_t, oil_p = (truth == ut.OIL) & lab, (pred == ut.OIL) & lab
        acc = 100 * (pred[lab] == truth[lab]).mean()
        msg = f"labelled-pixel accuracy {acc:.0f}% | real oil {int(oil_t.sum())} px, found {int((oil_t & oil_p).sum())} px"
    else:
        msg = "no real labels for this crop"

    n = 3 if have_truth else 2
    fig, ax = plt.subplots(1, n, figsize=(4.6 * n, 5.2))
    ax[0].imshow(rgb); ax[0].set_title("Satellite (true colour)")
    shown = pred.copy()
    if have_truth:
        shown = np.where(truth > 0, pred, 0)                       # compare only where experts labelled
    ax[1].imshow(paint(shown), interpolation="nearest"); ax[1].set_title("U-Net prediction")
    if have_truth:
        ax[2].imshow(paint(truth), interpolation="nearest"); ax[2].set_title("Real labels (experts)")
    for a in ax:
        a.axis("off")
    present = sorted(set(np.unique(shown)) | (set(np.unique(truth)) if have_truth else set()) - {0})
    handles = [Patch(color=colors[c - 1], label=ut.CLASS_NAMES[c - 1]) for c in present if c > 0]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, fontsize=9)
    fig.suptitle(f"{scene}, crop {crop}  |  {msg}", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    out = f"{OUT}/{scene}_crop{crop}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out, "|", msg, flush=True)
print(f"\nDone. Open the {OUT} folder and show these pictures.")