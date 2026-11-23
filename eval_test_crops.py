"""
eval_test_crops.py
Runs the saved model on every crop of the TEST split (from the stacked images)
and prints the overall score. Compare with the CSV result (mIoU 0.429, oil IoU 0.307).
Run from the Bluewatch folder:  py eval_test_crops.py
"""
import os
import warnings
import numpy as np
import rasterio
import joblib
from predict_image import predict_scene, CLASS_NAMES

warnings.filterwarnings("ignore")

ROOT = "MADOS/MADOS_nearest"
model = joblib.load("bluewatch_rf_final.joblib")
test_crops = open(ROOT + "/splits/test_X.txt").read().split()

cm = np.zeros((16, 16), dtype=np.int64)   # rows = truth, cols = prediction (index = class id)

for i, name in enumerate(test_crops, 1):
    scene, crop = name.rsplit("_", 1)
    img_path = f"{ROOT}/{scene}/{scene}_L2R_rhorc_{crop}.tif"
    lab_path = f"{ROOT}/{scene}/{scene}_L2R_cl_{crop}.tif"
    if not (os.path.exists(img_path) and os.path.exists(lab_path)):
        continue
    pred = predict_scene(img_path, model)
    with rasterio.open(lab_path) as src:
        truth = src.read(1)
    m = truth > 0
    np.add.at(cm, (truth[m].astype(int), pred[m].astype(int)), 1)
    if i % 10 == 0:
        print(f"{i}/{len(test_crops)} crops done")

tp = np.diag(cm).astype(float)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
iou = tp / np.maximum(tp + fp + fn, 1)
present = cm.sum(axis=1) > 0
present[0] = False

print("\nTEST SPLIT, scored from the stacked images")
print("labelled pixels:", int(cm.sum()))
print("mIoU:", round(iou[present].mean(), 3))
print("oil IoU:", round(iou[6], 3),
      "| oil recall:", round(tp[6] / max(tp[6] + fn[6], 1), 3),
      "| oil precision:", round(tp[6] / max(tp[6] + fp[6], 1), 3))
print("\nper-class IoU:")
for c in range(1, 16):
    if present[c]:
        print(f"  {CLASS_NAMES[c]}: {iou[c]:.2f}")