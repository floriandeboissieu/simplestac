# %% [md]
# # Download data
# [Source code](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples/download_data.py)
#
# This script aims at downloading the data necessary for the example.
# 
# The notebooks data are:
# 
# - a time series of Sentinel 2 data clip
# over a small region of interest (172MB)
# - a vector file of the region of interest (~100kB)
# 
# The following code download the data and stores them in the data folder.
# The downloaded data is 
# The destination folder is the one from which the script is executed,
# in which you should find:
# 
# - data / s2_scenes
# - data / roi.geojson

# %%
import geopandas as gpd
from path import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import zipfile

data_dir = Path(__file__).parent / "data"
s2_dir = data_dir / "s2_scenes"
roi_file = data_dir / "roi.geojson"

if not (s2_dir.exists() and roi_file.exists()):
    data_dir.mkdir_p()
    data_url = Path("https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip")
    
    with TemporaryDirectory(prefix="simplestac_") as tmpdir:
        # download and extraction directory (removed automatically after transfer)
        dl_dir = Path(tmpdir)

        s2_tmp_dir = dl_dir / "fordead_data-main/sentinel_data/dieback_detection_tutorial/study_area"
        roi_tmp_file = dl_dir / "fordead_data-main/vector/area_interest.shp"
        zip_path, _ = urlretrieve(data_url, dl_dir / data_url.name)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(dl_dir)
        s2_tmp_dir.move(data_dir).rename(s2_dir)
        gpd.read_file(roi_tmp_file).to_file(roi_file)
