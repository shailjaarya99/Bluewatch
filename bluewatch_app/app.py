"""
BlueWatch web app (runs on your own computer).

    py app.py

It opens http://127.0.0.1:5000 in your browser.
It looks for these next to the app, or one folder above it:
    bluewatch_rf_final.joblib          the trained model        (required)
    MADOS/MADOS_nearest/               stacked scenes           (for the sample list)
    test.csv                           pixel table              (for the "try test.csv" button)
"""
import base64
import json
import os
import re
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from flask import Flask, Response, jsonify, request, send_from_directory
from rasterio.io import MemoryFile

try:                                   # the U-Net needs PyTorch (installed for Python 3.12 on your PC)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
SEARCH = [Path.cwd(), HERE, HERE.parent]

BANDS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]
OIL = 6
MAX_PIXELS = 1_200_000      # biggest scene we accept (about 1100 x 1100)
MAX_CSV_ROWS = 400_000
TEST_CSV_ROWS = 50_000

CLASS_NAMES = {
    1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Floating Algae",
    4: "Natural Organic Material", 5: "Ship", 6: "Oil Spill", 7: "Marine Water",
    8: "Sediment-Laden Water", 9: "Foam", 10: "Turbid Water", 11: "Shallow Water",
    12: "Waves & Wakes", 13: "Oil Platform", 14: "Jellyfish", 15: "Sea snot",
}


def find(*parts):
    for base in SEARCH:
        p = base.joinpath(*parts)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- model
MODEL_PATH = find("bluewatch_rf_final.joblib") or find("bluewatch_rf.joblib")
MODEL = None
FEATURES, CLASSES, OIL_COL = [], np.arange(1, 16), None
if MODEL_PATH is not None:
    try:
        MODEL = joblib.load(MODEL_PATH)
        FEATURES = list(getattr(MODEL, "feature_names_in_",
                                BANDS + ["ndwi", "mndwi", "ndvi", "nir_swir", "fai"]))
        CLASSES = np.array(MODEL.classes_).astype(int)
        OIL_COL = list(CLASSES).index(OIL)
        print(f"Random Forest: {MODEL_PATH.name} | {len(FEATURES)} features | {len(CLASSES)} classes")
    except Exception as e:
        MODEL = None
        print(f"The Random Forest file was found but could not be loaded: {e}")


# ---------------------------------------------------------------- U-Net (only if PyTorch is installed)
UNET_PATH = find("bluewatch_unet.pt")
UNET = None
UNETS, UNET_NAMES = [], []
UDEV = "cpu"
UNET_MAX_PIXELS = 512 * 512
U_MEAN = np.array([0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
                   0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944], dtype=np.float32)
U_STD = np.array([0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
                  0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], dtype=np.float32)

if HAVE_TORCH:
    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):          # same layout as unet_train.py
        def __init__(self, cin=11, ncls=15, c=32):
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

    def unet_files():
        found = {}
        for base in SEARCH:
            for pat in ("bluewatch_unet.pt", "bluewatch_unet_v2_seed*.pt", "bluewatch_unet_v3_seed*.pt"):
                for p in sorted(base.glob(pat)):
                    found.setdefault(p.name, p)
        return list(found.values())

    UDEV = "cuda" if torch.cuda.is_available() else "cpu"
    for p in unet_files():
        try:
            sd = torch.load(p, map_location=UDEV)
            m = UNet(c=sd["e1.net.0.weight"].shape[0]).to(UDEV)
            m.load_state_dict(sd)
            m.eval()
            UNETS.append(m)
            UNET_NAMES.append(p.name)
        except Exception as e:
            print(f"Skipped {p.name}: {e}")
    if UNETS:
        UNET = UNETS[0]
        print(f"U-Net ensemble: {len(UNETS)} model(s) on {UDEV}: {', '.join(UNET_NAMES)}")

if MODEL is None and UNET is None:
    sys.exit("No model could be loaded.\n"
             "Put bluewatch_rf_final.joblib (and bluewatch_unet.pt) in your Bluewatch folder, then run the app again.")


def unet_scene(img):
    """img: (11, H, W) reflectance, NaN allowed. Returns oil probability and best non-oil class per pixel."""
    x = img.astype(np.float32)
    x = np.where(np.isfinite(x), x, U_MEAN[:, None, None])
    x = np.clip((x - U_MEAN[:, None, None]) / U_STD[:, None, None], -10, 40)
    h, w = x.shape[1:]
    t = torch.from_numpy(x)[None].to(UDEV)
    ph, pw = (-h) % 8, (-w) % 8                 # the U-Net halves the size 3 times
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        p = (sum(torch.softmax(m(t), 1) for m in UNETS) / len(UNETS))[0, :, :h, :w].float().cpu().numpy()
    q = p.copy()
    q[OIL - 1] = -1
    return p[OIL - 1].reshape(-1), (q.argmax(0) + 1).reshape(-1)


def make_features(b):
    """Same maths as training: 11 bands + 5 spectral indices."""
    d = b.copy()
    d["ndwi"] = (b.b3 - b.b8) / (b.b3 + b.b8 + 1e-8)
    d["mndwi"] = (b.b3 - b.b11) / (b.b3 + b.b11 + 1e-8)
    d["ndvi"] = (b.b8 - b.b4) / (b.b8 + b.b4 + 1e-8)
    d["nir_swir"] = b.b8 / b.b11.clip(lower=0.001)
    d["fai"] = b.b8 - (b.b4 + (b.b11 - b.b4) * (832.8 - 664.6) / (1613.7 - 664.6))
    return d[FEATURES]


def predict_proba(pixels, chunk=200_000):
    out = np.empty((len(pixels), len(CLASSES)), dtype=np.float32)
    for i in range(0, len(pixels), chunk):
        df = pd.DataFrame(pixels[i:i + chunk], columns=BANDS)
        out[i:i + chunk] = MODEL.predict_proba(make_features(df))
    return out


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def true_colour(img):
    """bands 4,3,2 (red, green, blue) stretched so dark sea scenes are visible"""
    rgb = np.stack([img[3], img[2], img[1]], -1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.8
    return (rgb * 255).astype(np.uint8)


def read_tif(data):
    with MemoryFile(data) as mf:
        with mf.open() as src:
            return src.read()


def score(truth, pred):
    """truth, pred: int arrays of class ids (1..15)."""
    cm = np.zeros((16, 16), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    present = cm.sum(1) > 0
    present[0] = False
    non_oil_truth = cm[1:].sum() - cm[OIL].sum()
    return {
        "labelled": int(cm[1:].sum()),
        "accuracy": float(tp[1:].sum() / max(cm[1:].sum(), 1)),
        "miou": float(iou[present].mean()) if present.any() else None,
        "oil_real": int(cm[OIL].sum()),
        "oil_iou": float(iou[OIL]) if cm[OIL].sum() or fp[OIL] else None,
        "oil_recall": float(tp[OIL] / (tp[OIL] + fn[OIL])) if cm[OIL].sum() else None,
        "oil_precision": float(tp[OIL] / (tp[OIL] + fp[OIL])) if (tp[OIL] + fp[OIL]) else None,
        "false_alarm_rate": float(fp[OIL] / non_oil_truth) if non_oil_truth else None,
        "false_alarms": int(fp[OIL]),
        "per_class": [
            {"id": c, "name": CLASS_NAMES[c], "real": int(cm[c].sum()),
             "predicted": int(cm[:, c].sum()), "iou": float(iou[c]) if present[c] else None}
            for c in range(1, 16)
        ],
    }


# ---------------------------------------------------------------- app
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


DEFAULT_LOGO = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#FBE4D8'/>
<stop offset='1' stop-color='#854F6C'/></linearGradient></defs>
<rect width='64' height='64' rx='16' fill='#190019'/>
<path d='M13 35c0-12 8-20 19-20s19 8 19 20q-3.170 5-6.330 0q-3.170 5-6.330 0q-3.170 5-6.330 0q-3.170 5-6.330 0q-3.170 5-6.330 0q-3.170 5-6.330 0z' fill='url(#g)'/>
<g stroke='#DFB6B2' stroke-width='2.2' stroke-linecap='round' fill='none'>
<path d='M21 39c-2 6 2 8 0 14'/><path d='M32 39c2 6-2 8 0 16'/><path d='M43 39c-2 6 2 8 0 14'/></g></svg>"""


@app.get("/logo")
def logo():
    """Your logo: put a file called logo.png (or .svg/.jpg/.webp) in the static folder."""
    for ext in ("png", "svg", "jpg", "jpeg", "webp", "ico"):
        p = HERE / "static" / f"logo.{ext}"
        if p.exists():
            return send_from_directory(p.parent, p.name)
    return Response(DEFAULT_LOGO, mimetype="image/svg+xml")


@app.get("/api/status")
def api_status():
    return jsonify(rf=MODEL is not None, unet=UNET is not None, device=UDEV, n_unets=len(UNETS),
                   unet_hint=bool(UNET is None and UNET_PATH is not None and not HAVE_TORCH),
                   samples=find("MADOS", "MADOS_nearest", "splits", "test_X.txt") is not None,
                   test_csv=find("test.csv") is not None,
                   custom_logo=any((HERE / "static" / f"logo.{e}").exists()
                                   for e in ("png", "svg", "jpg", "jpeg", "webp", "ico")))


# ---------------------------------------------------------------- samples
_samples = {"data": None}


def build_samples():
    root = find("MADOS", "MADOS_nearest")
    if root is None or not (root / "splits" / "test_X.txt").exists():
        return None
    cache = HERE / "samples_cache.json"
    names = (root / "splits" / "test_X.txt").read_text().split()
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("n") == len(names):
                return c["rows"]
        except Exception:
            pass
    rows = []
    for nm in names:
        scene, crop = nm.rsplit("_", 1)
        ip = root / scene / f"{scene}_L2R_rhorc_{crop}.tif"
        lp = root / scene / f"{scene}_L2R_cl_{crop}.tif"
        if not (ip.exists() and lp.exists()):
            continue
        with rasterio.open(lp) as s:
            lab = s.read(1)
        rows.append({"id": nm, "scene": scene, "crop": crop,
                     "oil": int((lab == OIL).sum()), "water": int((lab == 7).sum()),
                     "labelled": int((lab > 0).sum())})
    try:
        cache.write_text(json.dumps({"n": len(names), "rows": rows}))
    except Exception:
        pass
    return rows


@app.get("/api/samples")
def api_samples():
    if _samples["data"] is None:
        _samples["data"] = build_samples()
    rows = _samples["data"]
    if rows is None:
        return jsonify(available=False, oil=[], clean=[])
    oil = [r for r in rows if r["oil"] > 0]
    oil.sort(key=lambda r: (min(r["oil"], r["water"]) > 0, min(r["oil"], r["water"]), r["oil"]), reverse=True)
    clean = [r for r in rows if r["oil"] == 0 and r["labelled"] >= 800]
    clean.sort(key=lambda r: r["labelled"], reverse=True)
    return jsonify(available=True, oil=oil[:14], clean=clean[:14])


# ---------------------------------------------------------------- scene scan
def bad(msg, code=400):
    return jsonify(error=msg), code


@app.post("/api/predict")
def api_predict():
    t0 = time.time()
    note = None
    truth = None
    sid = request.form.get("sample")
    which = request.form.get("model") or ("unet" if UNET is not None else "rf")
    if which == "unet" and UNET is None:
        return bad("The U-Net is not loaded. It needs PyTorch (start the app with py -3.12 bluewatch_app/app.py) and the file bluewatch_unet.pt in your Bluewatch folder.")
    if which == "rf" and MODEL is None:
        return bad("The Random Forest is not loaded.")

    if sid:
        if not re.fullmatch(r"Scene_\d+_\d+", sid):
            return bad("That sample name is not valid.")
        root = find("MADOS", "MADOS_nearest")
        if root is None:
            return bad("The MADOS_nearest folder was not found next to the app.")
        scene, crop = sid.rsplit("_", 1)
        ip = root / scene / f"{scene}_L2R_rhorc_{crop}.tif"
        lp = root / scene / f"{scene}_L2R_cl_{crop}.tif"
        if not ip.exists():
            return bad(f"{ip.name} was not found.")
        img = read_tif(ip.read_bytes())
        if lp.exists():
            truth = read_tif(lp.read_bytes())[0]
        name = f"{scene.replace('_', ' ')}, crop {crop}"
    else:
        img_f = lab_f = None
        for f in request.files.getlist("files"):
            low = (f.filename or "").lower()
            if "_cl_" in low:
                lab_f = f
            elif "_conf_" in low or "_rep_" in low:
                continue
            elif img_f is None:
                img_f = f
        if img_f is None:
            return bad("Add the scene file. It is named like Scene_12_L2R_rhorc_3.tif (the 11-band one).")
        try:
            img = read_tif(img_f.read())
            if lab_f is not None:
                truth = read_tif(lab_f.read())[0]
        except Exception as e:
            return bad(f"Could not read that file as a GeoTIFF: {e}")
        name = img_f.filename

    if img.shape[0] != len(BANDS):
        return bad(f"This file has {img.shape[0]} bands. BlueWatch needs the stacked 11-band scene "
                   f"(files named ..._rhorc_N.tif inside MADOS_nearest).")
    n, h, w = img.shape
    if h * w > MAX_PIXELS:
        return bad(f"This scene is {w} x {h} pixels. The limit is about 1,100 x 1,100. Cut it into smaller tiles first.")
    if which == "unet" and h * w > UNET_MAX_PIXELS:
        return bad(f"The U-Net takes scenes up to 512 x 512 pixels, and this one is {w} x {h}. "
                   f"Switch to the Random Forest for big scenes, or cut the scene into tiles.")
    img = img.astype(np.float64)
    top = np.nanmax(img) if np.isfinite(img).any() else 0
    if top > 5:
        img = img / 10000.0
        note = "Values looked like 0-10000 digital numbers, so they were divided by 10,000."
    img_nan = img                         # the U-Net fills gaps itself
    img = np.nan_to_num(img)
    if truth is not None and truth.shape != (h, w):
        truth, note = None, "The label file is a different size from the scene, so it was ignored."

    if which == "unet":
        oilf, nonoil = unet_scene(img_nan)
        oilp = np.rint(oilf * 255).astype(np.uint8)
        nonoil = nonoil.astype(np.uint8)
    else:
        proba = predict_proba(img.reshape(n, -1).T)
        oilp = np.rint(proba[:, OIL_COL] * 255).astype(np.uint8)
        masked = proba.copy()
        masked[:, OIL_COL] = -1
        nonoil = CLASSES[masked.argmax(1)].astype(np.uint8)

    return jsonify(
        name=name, w=int(w), h=int(h), note=note, model=which, seconds=round(time.time() - t0, 2),
        oilp=b64(oilp), nonoil=b64(nonoil), rgb=b64(true_colour(img)),
        truth=b64(truth.astype(np.uint8)) if truth is not None else None,
    )



# ---------------------------------------------------------------- demo helpers
@app.get("/api/sample_csv")
def api_sample_csv():
    """A small pixel table cut from your test.csv, so the CSV check can be tried without hunting for a file."""
    p = find("test.csv")
    if p is None:
        return bad("test.csv was not found next to the app.", 404)
    df = pd.read_csv(p)
    df = df.sample(min(2000, len(df)), random_state=7)
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=bluewatch_sample_pixels.csv"})


@app.get("/api/analysis")
def api_analysis():
    p = find("analysis", "analysis.json")
    if p is None:
        return jsonify(available=False)
    try:
        return jsonify(available=True, **json.loads(p.read_text()))
    except Exception:
        return jsonify(available=False)


@app.get("/analysis/<path:fname>")
def analysis_file(fname):
    d = find("analysis")
    if d is None:
        return bad("Run analysis_report.py first.", 404)
    return send_from_directory(d, fname)


# ---------------------------------------------------------------- csv check
@app.post("/api/predict_csv")
def api_predict_csv():
    t0 = time.time()
    if MODEL is None:
        return bad("The pixel-table check uses the Random Forest, and it is not loaded.")
    use_test = request.form.get("use_test_csv") == "1"
    try:
        if use_test:
            p = find("test.csv")
            if p is None:
                return bad("test.csv was not found next to the app.")
            df, name, cap = pd.read_csv(p), p.name, TEST_CSV_ROWS
        else:
            f = request.files.get("file")
            if f is None:
                return bad("Choose a CSV file first.")
            df, name, cap = pd.read_csv(f.stream), f.filename, MAX_CSV_ROWS
    except Exception as e:
        return bad(f"Could not read that CSV: {e}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [b for b in BANDS if b not in df.columns]
    if missing:
        return bad("This CSV is missing columns: " + ", ".join(missing) +
                   ". It needs b1 to b12 (like train.csv).")
    total = len(df)
    sampled = total > cap
    if sampled:
        df = df.sample(cap, random_state=42)

    pix = np.nan_to_num(df[BANDS].to_numpy(dtype=np.float64))
    if pix.max() > 5:
        pix = pix / 10000.0
    proba = predict_proba(pix)
    pred = CLASSES[proba.argmax(1)]
    counts = np.bincount(pred, minlength=16)[1:16].tolist()

    out = {"name": name, "rows": int(len(df)), "total_rows": int(total), "sampled": bool(sampled),
           "counts": counts, "seconds": round(time.time() - t0, 2), "scores": None}
    if "label" in df.columns:
        truth = pd.to_numeric(df["label"], errors="coerce").round().fillna(0).astype(int).to_numpy()
        ok = (truth >= 1) & (truth <= 15)
        if ok.any():
            out["scores"] = score(truth[ok], pred[ok])
    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.3, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"BlueWatch is running at http://127.0.0.1:{port}  (press Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
