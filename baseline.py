import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score, f1_score

bands = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]
index_cols = ["ndwi", "mndwi", "ndvi", "nir_swir", "fai"]


def load(name):
    df = pd.read_csv(name + ".csv")
    df["label"] = df["label"].astype(int)
    df["ndwi"] = (df.b3 - df.b8) / (df.b3 + df.b8 + 1e-8)
    df["mndwi"] = (df.b3 - df.b11) / (df.b3 + df.b11 + 1e-8)
    df["ndvi"] = (df.b8 - df.b4) / (df.b8 + df.b4 + 1e-8)
    df["nir_swir"] = df.b8 / df.b11.clip(lower=0.001)
    df["fai"] = df.b8 - (df.b4 + (df.b11 - df.b4) * (832.8 - 664.6) / (1613.7 - 664.6))
    return df


train = load("train")          
test = load("test")             

for use_indices in [ True]:
    cols = bands + (index_cols if use_indices else [])
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(train[cols], train["label"])
    pred = model.predict(test[cols])

    labels = list(range(1, 16))
    iou = jaccard_score(test["label"], pred, labels=labels, average=None)
    dice = f1_score(test["label"], pred, labels=labels, average=None)

    print("indices used:", use_indices)
    print("mIoU:", round(iou.mean(), 3))
    print("oil IoU:", round(iou[5], 3), "| oil Dice:", round(dice[5], 3))
    print("per-class IoU:", iou.round(2))

from sklearn.metrics import confusion_matrix
import joblib

cm = confusion_matrix(test["label"], pred, labels=labels)
print("oil pixels predicted as (% per class 1-15):")
print((cm[5] / cm[5].sum() * 100).round(1))

imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
print(imp.round(3))

joblib.dump(model, "bluewatch_rf_final.joblib")