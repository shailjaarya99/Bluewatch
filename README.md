# BlueWatch: Marine Oil Spill Detection from Sentinel-2

BlueWatch finds oil on the sea surface in Sentinel-2 satellite images. It labels every pixel as one of 15 classes (oil spill, marine water, foam, algae, ships, debris and so on) using the **MADOS** dataset. Two model families are compared: a **Random Forest** (one pixel at a time) and a **U-Net** (whole 240 x 240 crops, so it sees neighbouring pixels). A small Flask web app lets you try the models on test scenes.

## Dataset

- **MADOS** (Marine Debris and Oil Spill), Kikaki et al., ISPRS Journal 2024. Zenodo record 10664073, licence CC BY 4.0.
- 174 Sentinel-2 scenes, 2,803 crops of 240 x 240 pixels, 11 bands, 15 classes.
- Only about 1.48 million pixels (~0.9%) are labelled. Labels are sparse.
- The official train / validation / test split is used (1,433 / 642 / 728 crops).
- The dataset is **not included** in this repository because of its size. Download it from Zenodo and put it in a `MADOS/` folder.

## Results (test set, 361,147 labelled pixels)

| Model | Pixel acc. | mIoU | Oil IoU | Oil recall | Oil precision |
|---|---|---|---|---|---|
| Guess "all water" | 38.6% | - | - | - | - |
| Random Forest | not measured | 0.429 | 0.307 | 82.4% | 32.9% |
| U-Net, first run | 80.8% | 0.573 | 0.386 | 61.1% | 51.1% |
| U-Net + Dice, mean of 3 seeds | 82.6% | 0.570 | 0.436 | 69.0% | 54.1% |
| Ensemble of 5 U-Nets | 82.4% | 0.616 | 0.427 | 66.8% | 54.2% |

Honest reading of these numbers:

- A pixel accuracy of 90% was **not reached**. The best single model got 83.7%. Validation scores are higher than test scores, and that gap, not the model size, is the main limit.
- About two thirds of real oil pixels are found, but only about half of the pixels flagged as oil are really oil. Oil is mostly confused with ordinary marine water.
- MADOS has an unrealistically high oil share (~18% of labelled pixels). In real use, precision would be lower.
- The confidence slider in the app barely changes precision between 20% and 80%.

## Files

| File | What it does |
|---|---|
| `data_extraction.py`, `stack_patches.py`, `spectral_utils.py` | Build the pixel tables and the spectral indices |
| `baseline.py`, `model_tuning.py` | Random Forest baseline and tuning |
| `unet_train.py`, `unet_train_v2.py`, `unet_train_v3.py` | U-Net training (v2 adds Dice loss and seeds, v3 is the wider model) |
| `unet_report.py`, `eval_test_crops.py` | Score a saved U-Net on the test crops |
| `ensemble_eval.py` | Average several U-Nets and score the ensemble |
| `analysis_report.py` | Confusion matrix, precision-recall curve, failure cases (writes `analysis/`) |
| `make_demo_images.py`, `find_oil_crops.py`, `check_scene.py`, `predict_image.py` | Demo pictures and helpers |
| `bluewatch_app/` | Flask + JavaScript web app |
| `*.pt` | Trained U-Net weights |
| `unet_metrics*.json`, `results.txt`, `final_results.txt` | Measured results |

## How to run

Python 3.12, PyTorch, scikit-learn, rasterio, numpy, matplotlib, Flask.

```bash
pip install torch scikit-learn rasterio numpy matplotlib flask

# 1. download MADOS from Zenodo (record 10664073) into ./MADOS
# 2. train a U-Net
python unet_train.py

# 3. score it and the ensemble
python unet_report.py
python ensemble_eval.py

# 4. make the analysis charts used by the app
python analysis_report.py

# 5. start the web app
python bluewatch_app/app.py     # opens http://127.0.0.1:5000
```

The Random Forest model file (`.joblib`, ~400 MB) and `train.csv` are not in the repository. Re-create them with `data_extraction.py` and `baseline.py`.

## Limitations

- Sparse labels and one sensor family (Sentinel-2).
- Validation and test scores differ by several mIoU points, so single numbers are noisy.
- Not tested on real, independent oil-spill events outside MADOS. This is a student project and not an operational monitoring system.

## Credits

Data: Kikaki, Kakogeorgiou, Hoteit, Karantzalos, *Detecting marine pollutants and sea surface features with deep learning in Sentinel-2 imagery*, ISPRS Journal of Photogrammetry and Remote Sensing, 2024 (MADOS, CC BY 4.0). U-Net: Ronneberger et al., 2015.
