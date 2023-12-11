from pandas import to_datetime
from path import Path
from simplestac.utils import ItemCollection
from simplestac.local import build_item_collection, collection_format
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import xarray as xr
import zipfile

tmpdir = Path(TemporaryDirectory(prefix="simplestac_").name)
tmpdir = Path("/tmp/simplestac_szlnywhw")
print(tmpdir) # to keep track of the directory to remove
data_dir = tmpdir/'fordead_data-main'

if not data_dir.exists():
    data_url = Path("https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip")

    with TemporaryDirectory() as tmpdir2:
        dl_dir = Path(tmpdir2)
        zip_path, _ = urlretrieve(data_url, dl_dir / data_url.name)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmpdir.mkdir_p())

image_dir = data_dir / "sentinel_data/dieback_detection_tutorial/study_area"

# make a temporary directory where the catalog will be written
coll_path = tmpdir / "collection.json"

import cProfile
proffile = Path('/tmp/test.prof').remove_p()
cProfile.run("build_item_collection(image_dir, collection_format('S2_L2A_THEIA'))", filename=proffile)
print(proffile)



profiler = cProfile.Profile()
proffile = Path('/tmp/test.prof').remove()
fmt = collection_format('S2_L2A_THEIA')
with cProfile.Profile() as profiler:
    build_item_collection(image_dir, fmt, progress=True, validate=False)
    profiler.create_stats()
    profiler.dump_stats(proffile)
print(proffile)