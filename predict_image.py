import sys
import numpy as np
import pandas as pd
import rasterio
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BANDS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]
INDEX_COLS = ["ndwi", "mndwi", "ndvi", "nir_swir", "fai"]

# official MADOS classes (from utils/assets.py in the MADOS repo)
CLASS_NAMES = {
    1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Floating Algae",
    4: "Natural Organic Material", 5: "Ship", 6: "Oil Spill", 7: "Marine Water",
    8: "Sediment-Laden Water", 9: "Foam", 10: "Turbid Water", 11: "Shallow Water",
    12: "Waves & Wakes", 13: "Oil Platform", 14: "Jellyfish", 15: "Sea snot",
}
# colours: oil is the loud one, water is calm
CLASS_COLORS = {
    1: "#ff00ff", 2: "#00a650", 3: "#8dd35f", 4: "#8c5a2b", 5: "#ffe600",
    6: "#ff3b00", 7: "#10243d", 8: "#b07a3c", 9: "#ffffff", 10: "#4aa3c7",
    11: "#2ec4b6", 12: "#9aa5b1", 13: "#ff9f1c", 14: "#c77dff", 15: "#6b705c",
}


def add_indices(df):
    df["ndwi"] = (df.b3 - df.b8) / (df.b3 + df.b8 + 1e-8)
    df["mndwi"] = (df.b3 - df.b11) / (df.b3 + df.b11 + 1e-8)
    df["ndvi"] = (df.b8 - df.b4) / (df.b8 + df.b4 + 1e-8)
    df["nir_swir"] = df.b8 / df.b11.clip(lower=0.001)
    df["fai"] = df.b8 - (df.b4 + (df.b11 - df.b4) * (832.8 - 664.6) / (1613.7 - 664.6))
    return df


def predict_scene(tif_path, model):
    with rasterio.open(tif_path) as src:
        img = src.read()
    n, h, w = img.shape
    if n != len(BANDS):
        raise ValueError(f"expected {len(BANDS)} bands, got {n}")
    df = add_indices(pd.DataFrame(img.reshape(n, h * w).T, columns=BANDS))
    pred = model.predict(df[BANDS + INDEX_COLS]).reshape(h, w)
    return pred


def to_rgb_map(class_map):
    """class ids -> RGB image. 0 (unlabeled) stays black."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cid, hexcol in CLASS_COLORS.items():
        rgb[class_map == cid] = matplotlib.colors.to_rgb(hexcol)
    return rgb


def run(tif_path, model_path="bluewatch_rf_final.joblib", out_path="classified_output.png"):
    model = joblib.load(model_path)
    pred = predict_scene(tif_path, model)

    label_path = tif_path.replace("_rhorc_", "_cl_")
    truth = None
    try:
        with rasterio.open(label_path) as src:
            truth = src.read(1)
    except Exception:
        pass

    # --- numbers ---
    print("\nPredicted class breakdown (% of pixels):")
    vals, cnts = np.unique(pred, return_counts=True)
    for v, c in sorted(zip(vals, cnts), key=lambda x: -x[1]):
        print(f"  {CLASS_NAMES.get(int(v), v)}: {100 * c / pred.size:.2f}%")

    if truth is not None:
        mask = truth > 0                      # 0 = not annotated
        if mask.any():
            acc = (pred[mask] == truth[mask]).mean()
            print(f"\nPixel accuracy on labelled pixels: {100 * acc:.1f}%  ({mask.sum()} labelled pixels)")
            for cid in sorted(set(np.unique(truth[mask])) | {6}):
                t = (truth == cid) & mask
                p = (pred == cid) & mask
                union = (t | p).sum()
                if union:
                    print(f"  IoU {CLASS_NAMES.get(int(cid), cid)}: {(t & p).sum() / union:.2f}")

    # --- picture ---
    panels = 2 if truth is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 6), facecolor="#111111")
    axes = np.atleast_1d(axes)
    axes[0].imshow(to_rgb_map(pred), interpolation="nearest")
    axes[0].set_title("BlueWatch prediction", color="white")
    if truth is not None:
        axes[1].imshow(to_rgb_map(truth), interpolation="nearest")
        axes[1].set_title("Real labels (black = not annotated)", color="white")
    for ax in axes:
        ax.axis("off")

    present = sorted(set(np.unique(pred)) | (set(np.unique(truth)) if truth is not None else set()))
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in present if c in CLASS_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               labelcolor="white", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    print("\nsaved:", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: py predict_image.py path/to/Scene_X_L2R_rhorc_N.tif")
        sys.exit(1)
    run(sys.argv[1])