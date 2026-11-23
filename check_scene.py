import glob
import rasterio

files = glob.glob("MADOS/MADOS_nearest/Scene_0/*.tif")
for f in files:
    print(f)

stacked = [f for f in files if "_cl_" not in f and "_conf_" not in f and "_rep_" not in f][0]
with rasterio.open(stacked) as src:
    img = src.read()
    print("\nchecking:", stacked)
    print("bands, height, width:", img.shape)
    print("min / max:", img.min(), img.max())   