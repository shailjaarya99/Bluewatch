BlueWatch web app

1. Put this folder inside your Bluewatch folder, so you have:
      Bluewatch/bluewatch_app/app.py
      Bluewatch/bluewatch_rf_final.joblib      (Random Forest)
      Bluewatch/bluewatch_unet.pt              (U-Net, made by unet_train.py)
      Bluewatch/MADOS/MADOS_nearest/...        (for the sample scenes)
      Bluewatch/test.csv                       (for the "run on test.csv" button)

2. Install Flask once, for the Python that has PyTorch (3.12):
      py -3.12 -m pip install flask

3. Start it, from the Bluewatch folder:
      py -3.12 bluewatch_app/app.py      both models (U-Net + Random Forest)
      py bluewatch_app/app.py            Random Forest only (needs: py -m pip install flask)
   Your browser opens at http://127.0.0.1:5000. Press Ctrl+C in the terminal to stop.

Your logo: save it as  bluewatch_app/static/logo.png  (svg, jpg and webp also work).
It becomes the tab icon and the logo in the top bar. Refresh the page after adding it.

The numbers in "How far to trust it" come from  static/report.json.
Edit that file after you retrain.


--- Update (v2 page) ---
One working page: choose a scene, read the result, then the "How good is it?" tabs.
It averages every U-Net file it finds next to the app (bluewatch_unet.pt, bluewatch_unet_v2_seed*.pt, bluewatch_unet_v3_seed*.pt).
The "Oil mistakes" and "Failures" tabs fill in after you run, in your Bluewatch folder:
    py -3.12 analysis_report.py
Start the app from the Bluewatch folder:   py -3.12 bluewatch_app/app.py
The CSV check is under "Advanced": it first gives you a sample CSV to download, then you choose it.
