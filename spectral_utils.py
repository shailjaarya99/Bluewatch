"""
spectral_utils.py
The maths and physics used in the oil spill project.
Works on numpy arrays (single pixels, columns of a table, or whole images).
"""
import numpy as np

# Sentinel-2A band centre wavelengths in nanometres (physics: where each band "looks")
WAVELENGTH_NM = {
    "B2": 492.4,    # blue
    "B3": 559.8,    # green
    "B4": 664.6,    # red
    "B8": 832.8,    # NIR
    "B8A": 864.7,   # narrow NIR
    "B11": 1613.7,  # SWIR-1
    "B12": 2202.4,  # SWIR-2
}

EPS = 1e-8  # avoids division by zero


# ---------- 1. Radiometry ----------
def to_reflectance(dn, scale=10000.0, offset=0.0):
    """
    Sentinel-2 stores reflectance as integers: reflectance = (DN - offset) / 10000.
    Newer products (processing baseline >= 04.00) have offset = 1000. Check your data!
    """
    refl = (np.asarray(dn, dtype=np.float32) - offset) / scale
    return np.clip(refl, 0.0, 1.0)


# ---------- 2. Spectral indices (all are "normalised differences" or ratios) ----------
def normalized_difference(a, b):
    """(a - b) / (a + b): range -1..1, cancels out overall brightness."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a - b) / (a + b + EPS)


def ndwi(green, nir):
    """McFeeters NDWI: water is positive, land/vegetation negative."""
    return normalized_difference(green, nir)


def mndwi(green, swir1):
    """Modified NDWI (Xu): uses SWIR instead of NIR."""
    return normalized_difference(green, swir1)


def ndvi(nir, red):
    """Vegetation index; high values flag floating algae / vegetation look-alikes."""
    return normalized_difference(nir, red)


def nir_swir_ratio(nir, swir1):
    """Simple band ratio. Oil and water absorb strongly in SWIR, so ratios shift."""
    return np.asarray(nir, dtype=np.float32) / (np.asarray(swir1, dtype=np.float32) + EPS)


def fai(red, nir, swir1):
    """
    Floating Algae Index (Hu, 2009). It draws a straight baseline between red and SWIR
    reflectance (linear interpolation in wavelength) and measures how far NIR sits above it.
        FAI = NIR - [ RED + (SWIR - RED) * (l_nir - l_red) / (l_swir - l_red) ]
    Floating material (algae, some oil emulsions, debris) gives a positive FAI.
    """
    l_red, l_nir, l_swir = WAVELENGTH_NM["B4"], WAVELENGTH_NM["B8"], WAVELENGTH_NM["B11"]
    baseline = red + (swir1 - red) * (l_nir - l_red) / (l_swir - l_red)
    return nir - baseline


def build_features(b2, b3, b4, b8, b11, b12):
    """Stack raw bands + indices into one (N, 10) feature matrix (or (H, W, 10) for images)."""
    feats = [
        b2, b3, b4, b8, b11, b12,
        ndwi(b3, b8),
        ndvi(b8, b4),
        nir_swir_ratio(b8, b11),
        fai(b4, b8, b11),
    ]
    return np.stack(feats, axis=-1)


FEATURE_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDWI", "NDVI", "NIR_SWIR", "FAI"]


def add_features_df(df):
    """
    Takes a DataFrame with columns b1..b12 (already reflectance) and adds index columns.
    NOTE: this dataset is already scaled to reflectance, so we do NOT divide by 10000.
    """
    out = df.copy()
    out["ndwi"] = ndwi(df["b3"].values, df["b8"].values)
    out["mndwi"] = mndwi(df["b3"].values, df["b11"].values)
    out["ndvi"] = ndvi(df["b8"].values, df["b4"].values)
    out["nir_swir"] = nir_swir_ratio(df["b8"].values, df["b11"].values)
    out["fai"] = fai(df["b4"].values, df["b8"].values, df["b11"].values)
    return out


# ---------- 3. Evaluation metrics ----------
def confusion_matrix(y_true, y_pred, n_classes):
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    idx = n_classes * y_true + y_pred
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_per_class(cm):
    """IoU = TP / (TP + FP + FN)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp / (tp + fp + fn + EPS)


def dice_per_class(cm):
    """Dice = 2TP / (2TP + FP + FN)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return 2 * tp / (2 * tp + fp + fn + EPS)


def miou(cm):
    """Mean IoU over classes that actually appear in the ground truth."""
    present = cm.sum(axis=1) > 0
    return iou_per_class(cm)[present].mean()


# ---------- quick self-test ----------
if __name__ == "__main__":
    g, n = np.array([0.08]), np.array([0.02])
    print("NDWI of clear water (should be > 0):", ndwi(g, n))
    y_t = np.array([0, 0, 1, 1, 1, 2])
    y_p = np.array([0, 1, 1, 1, 0, 2])
    cm = confusion_matrix(y_t, y_p, 3)
    print("confusion matrix:\n", cm)
    print("IoU:", iou_per_class(cm).round(3), "mIoU:", round(miou(cm), 3))
    print("Dice:", dice_per_class(cm).round(3))
