"""
unet_train.py
BlueWatch: U-Net on stacked 11-band Sentinel-2 crops (MADOS).

Run from the Bluewatch folder, with the Python that has PyTorch (3.12):
    py -3.12 unet_train.py           full run
    py -3.12 unet_train.py quick     tiny run, just to check everything works

What it does:
  1. loads the train / val / test crops listed in MADOS_nearest/splits
  2. trains a small U-Net (the net sees neighbouring pixels, not just one pixel)
  3. keeps the best model by validation mIoU  -> bluewatch_unet.pt
  4. scores that model once on the TEST crops and prints the result
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
ROOT = "MADOS/MADOS_nearest"
EPOCHS = 2 if QUICK else 30
BATCH = 16
LR = 2e-3
OIL = 6                       # class id of Oil Spill (labels are 1..15, 0 = not annotated)
NUM_CLASSES = 15
MODEL_FILE = "bluewatch_unet_quick.pt" if QUICK else "bluewatch_unet.pt"   # quick run never touches the real model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Marine Debris", "Dense Sargassum", "Sparse Floating Algae", "Natural Organic Material",
               "Ship", "Oil Spill", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water",
               "Shallow Water", "Waves & Wakes", "Oil Platform", "Jellyfish", "Sea snot"]

# per-band mean / std of MADOS (from the official repo, utils/dataset.py)
MEAN = np.array([0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
                 0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944], dtype=np.float32)
STD = np.array([0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
                0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], dtype=np.float32)

# how common each class is (official repo) -> used to weight rare classes up
CLASS_FREQ = np.array([0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 0.34775, 0.20638,
                       0.00062, 0.1169, 0.09188, 0.01309, 0.00917, 0.00176, 0.00963], dtype=np.float32)


# ------------------------------------------------------------------ data
def load_split(split_name, limit=None):
    names = open(f"{ROOT}/splits/{split_name}_X.txt").read().split()
    if limit:
        names = names[:limit]
    imgs, labs = [], []
    n_bad = 0
    for k, name in enumerate(names, 1):
        scene, crop = name.rsplit("_", 1)
        ip = f"{ROOT}/{scene}/{scene}_L2R_rhorc_{crop}.tif"
        lp = f"{ROOT}/{scene}/{scene}_L2R_cl_{crop}.tif"
        if not (os.path.exists(ip) and os.path.exists(lp)):
            continue
        with rasterio.open(ip) as s:
            x = s.read()
        with rasterio.open(lp) as s:
            y = s.read(1)
        if x.shape != (11, 240, 240) or y.shape != (240, 240):
            continue
        x = x.astype(np.float32)
        bad = ~np.isfinite(x)
        if bad.any():                                   # NaN / inf pixels would poison the whole model
            n_bad += 1
            x = np.where(bad, MEAN[:, None, None], x)   # unreadable pixel -> "average sea"
        x = np.clip((x - MEAN[:, None, None]) / STD[:, None, None], -10, 40).astype(np.float16)
        y = np.where((y >= 1) & (y <= 15), y.astype(np.int16) - 1, 255).astype(np.uint8)   # 0..14, 255 = ignore
        imgs.append(x)
        labs.append(y)
        if k % 200 == 0:
            print(f"  {split_name}: {k}/{len(names)} crops read", flush=True)
    print(f"{split_name}: {len(imgs)} crops loaded", flush=True)
    if n_bad:
        print(f"  {split_name}: fixed NaN/inf pixels in {n_bad} crops", flush=True)
    return np.stack(imgs), np.stack(labs)


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=11, ncls=NUM_CLASSES, c=32):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(cin, c), Block(c, 2 * c), Block(2 * c, 4 * c)
        self.mid = Block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), Block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), Block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), Block(2 * c, c)
        self.out = nn.Conv2d(c, ncls, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        c = self.d3(torch.cat([self.u3(m), c], 1))
        b = self.d2(torch.cat([self.u2(c), b], 1))
        a = self.d1(torch.cat([self.u1(b), a], 1))
        return self.out(a)


# ------------------------------------------------------------------ helpers
def augment(x, y):
    """random flips / 90-degree turns, same for image and label"""
    if np.random.rand() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if np.random.rand() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = np.random.randint(4)
    return torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))


@torch.no_grad()
def evaluate(model, X, Y):
    """confusion matrix over annotated pixels -> mIoU, per-class IoU, oil precision/recall"""
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(0, len(X), 32):
        x = torch.from_numpy(X[i:i + 32]).to(DEVICE).float()
        pred = model(x).argmax(1).cpu().numpy()
        y = Y[i:i + 32]
        m = y != 255
        np.add.at(cm, (y[m].astype(int), pred[m]), 1)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    present = cm.sum(1) > 0
    o = OIL - 1
    return dict(miou=iou[present].mean(), iou=iou, present=present,
                oil_iou=iou[o], oil_recall=tp[o] / max(tp[o] + fn[o], 1),
                oil_precision=tp[o] / max(tp[o] + fp[o], 1))


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("device:", DEVICE, "| quick mode:", QUICK, flush=True)
    t0 = time.time()
    lim = 24 if QUICK else None
    Xtr, Ytr = load_split("train", lim)
    Xva, Yva = load_split("val", lim)
    Xte, Yte = load_split("test", lim)
    print(f"data ready in {time.time() - t0:.0f}s\n", flush=True)

    model = UNet().to(DEVICE)
    w = 1.0 / np.sqrt(CLASS_FREQ)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=255)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(Xtr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t1 = time.time()
        order = np.random.permutation(len(Xtr))
        total, seen, skipped = 0.0, 0, 0
        for i in range(0, len(order), BATCH):
            idx = np.sort(order[i:i + BATCH])
            x = torch.from_numpy(Xtr[idx]).to(DEVICE).float()
            y = torch.from_numpy(Ytr[idx]).to(DEVICE).long()
            if (y != 255).sum() == 0:                   # nothing labelled in this batch
                continue
            x, y = augment(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, enabled=use_amp):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):                # a broken batch must never touch the model
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        if seen == 0:
            sys.exit("Every batch gave a broken loss, so training stopped. Send this message to Claude.")
        r = evaluate(model, Xva, Yva)
        flag = ""
        if r["miou"] > best:
            best = r["miou"]
            torch.save(model.state_dict(), MODEL_FILE)
            flag = "  <- saved"
        print(f"epoch {epoch:2d}/{EPOCHS} | loss {total / seen:.3f} | VAL mIoU {r['miou']:.3f} "
              f"oil IoU {r['oil_iou']:.3f} | {time.time() - t1:.0f}s{flag}"
              + (f" | skipped {skipped} bad batches" if skipped else ""), flush=True)

    print("\n=== FINAL: best model on the TEST crops ===")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    r = evaluate(model, Xte, Yte)
    print(f"TEST mIoU: {r['miou']:.3f}")
    print(f"TEST oil IoU: {r['oil_iou']:.3f} | oil recall: {r['oil_recall']:.3f} | "
          f"oil precision: {r['oil_precision']:.3f}")
    print("\nper-class IoU:")
    for c in range(NUM_CLASSES):
        if r["present"][c]:
            print(f"  {CLASS_NAMES[c]}: {r['iou'][c]:.2f}")
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")

    if not QUICK and np.isfinite(r["miou"]):   # the website reads this file to show the real U-Net score
        with open("unet_metrics.json", "w") as f:
            json.dump(dict(
                test_miou=round(float(r["miou"]), 3),
                oil_iou=round(float(r["oil_iou"]), 3),
                oil_recall=round(float(r["oil_recall"]), 3),
                oil_precision=round(float(r["oil_precision"]), 3),
                per_class={CLASS_NAMES[c]: round(float(r["iou"][c]), 2)
                           for c in range(NUM_CLASSES) if r["present"][c]},
                epochs=EPOCHS), f, indent=2)
        print("saved: bluewatch_unet.pt, unet_metrics.json")