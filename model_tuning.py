"""
model_tuning.py
Tries a few quick fixes for "model calls water oil".
  A) baseline, 16 features
  B) drop b1 (it was the top feature, which can mean it memorises scenes)
  C) B + simpler trees (min_samples_leaf, max_depth)
Then, for the best of them, sweeps an oil-probability threshold.

Choose the winner using the VAL numbers. TEST is printed only to report the final score.
Run:  py model_tuning.py          (full, takes a while)
      py model_tuning.py quick    (small, 1 minute, just to see it works)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
BANDS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]
IDX = ["ndwi", "mndwi", "ndvi", "nir_swir", "fai"]


def load(name):
    df = pd.read_csv(name + ".csv")
    df["label"] = df["label"].astype(int)
    df["ndwi"] = (df.b3 - df.b8) / (df.b3 + df.b8 + 1e-8)
    df["mndwi"] = (df.b3 - df.b11) / (df.b3 + df.b11 + 1e-8)
    df["ndvi"] = (df.b8 - df.b4) / (df.b8 + df.b4 + 1e-8)
    df["nir_swir"] = df.b8 / df.b11.clip(lower=0.001)
    df["fai"] = df.b8 - (df.b4 + (df.b11 - df.b4) * (832.8 - 664.6) / (1613.7 - 664.6))
    return df


def scores(y, pred):
    labels = np.arange(1, 16)
    cm = np.zeros((16, 16), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    present = cm.sum(1) > 0
    present[0] = False
    return dict(mIoU=iou[present].mean(), oil_IoU=iou[6],
                oil_recall=tp[6] / max(tp[6] + fn[6], 1),
                oil_precision=tp[6] / max(tp[6] + fp[6], 1))


train, val, test = load("train"), load("vali"), load("test")
if QUICK:
    train = train.sample(50000, random_state=42)
n_trees = 20 if QUICK else 100

variants = {
    "A baseline (16 features)": (BANDS + IDX, dict()),
    "B no b1 (15 features)": ([b for b in BANDS if b != "b1"] + IDX, dict()),
    "C no b1 + simpler trees": ([b for b in BANDS if b != "b1"] + IDX,
                                dict(min_samples_leaf=5, max_depth=25)),
}

results, models = [], {}
for name, (cols, extra) in variants.items():
    print("training:", name, flush=True)
    m = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced",
                               n_jobs=-1, random_state=42, **extra)
    m.fit(train[cols], train["label"])
    models[name] = (m, cols)
    sv = scores(val["label"].values, m.predict(val[cols]))
    st = scores(test["label"].values, m.predict(test[cols]))
    results.append((name, sv, st))

print("\n=== results (choose by VAL) ===")
print(f"{'variant':32s} {'VAL mIoU':>9s} {'VAL oil':>8s} | {'TEST mIoU':>9s} {'TEST oil':>8s}")
for name, sv, st in results:
    print(f"{name:32s} {sv['mIoU']:9.3f} {sv['oil_IoU']:8.3f} | {st['mIoU']:9.3f} {st['oil_IoU']:8.3f}")

best_name = max(results, key=lambda r: r[1]["oil_IoU"])[0]
m, cols = models[best_name]
print("\nbest by VAL oil IoU:", best_name)

print("\n=== oil threshold sweep for that model (pick t on VAL) ===")
classes = list(m.classes_)
oil_i = classes.index(6)


def predict_with_threshold(df, t):
    proba = m.predict_proba(df[cols])
    pred = np.array(classes)[proba.argmax(1)]
    # oil only if its probability is at least t; otherwise fall back to the best non-oil class
    not_oil = proba.copy()
    not_oil[:, oil_i] = -1
    fallback = np.array(classes)[not_oil.argmax(1)]
    return np.where((pred == 6) & (proba[:, oil_i] < t), fallback, pred)


print(f"{'t':>5s} | {'VAL oil IoU':>11s} {'prec':>6s} {'recall':>6s} | {'TEST oil IoU':>12s} {'prec':>6s} {'recall':>6s}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    sv = scores(val["label"].values, predict_with_threshold(val, t))
    st = scores(test["label"].values, predict_with_threshold(test, t))
    print(f"{t:5.1f} | {sv['oil_IoU']:11.3f} {sv['oil_precision']:6.2f} {sv['oil_recall']:6.2f} | "
          f"{st['oil_IoU']:12.3f} {st['oil_precision']:6.2f} {st['oil_recall']:6.2f}")