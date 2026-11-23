import glob, os
import rasterio

root = "MADOS/MADOS_nearest"
test_crops = set(open(root + "/splits/test_X.txt").read().split())

rows = []
for f in glob.glob(root + "/Scene_*/*_cl_*.tif"):
    scene = os.path.basename(os.path.dirname(f))
    crop = os.path.basename(f).split("_cl_")[-1].replace(".tif", "")
    with rasterio.open(f) as src:
        lab = src.read(1)
    oil = int((lab == 6).sum())
    water = int((lab == 7).sum())
    if oil > 0 and water > 0:
        rows.append((min(oil, water), oil, water, scene + "_" + crop in test_crops, scene, crop))

rows.sort(reverse=True)
print("min(oil,water) | oil | water | in test | scene | crop")
for r in rows[:15]:
    print(r)