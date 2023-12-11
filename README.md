# SimpleSTAC

SimpleSTAC eases the use of pystac.

# Install

__Recommendation:__ install the package in a virtual environment. See
[miniforge](https://github.com/conda-forge/miniforge) for conda/mamba, or 
[venv](https://docs.python.org/3/library/venv.html) for virtualenv.

Within a conda env:
```shell
mamba env create -n simplestac -f https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/raw/main/environment.yml
```

Within a virtualenv:
```shell
pip install git+https://forgemia.inra.fr/umr-tetis/stac/simplestac
```

# Examples

## Example data
Example data can be downloaded with the following:
```python
from path import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import zipfile

tmpdir = Path(TemporaryDirectory(prefix="simplestac_").name)
print(tmpdir) # to keep track of the directory to remove
data_dir = tmpdir/'fordead_data-main'

if not data_dir.exists():
    data_url = Path("https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip")

    with TemporaryDirectory() as tmpdir2:
        dl_dir = Path(tmpdir2)
        zip_path, _ = urlretrieve(data_url, dl_dir / data_url.name)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmpdir.mkdir_p())

```

__Before exiting your python session, don't forget to remove the temporary directory:__
```python
tmpdir.rmtree()
```

## Create a local STAC ItemCollection

In this example, local files (see previous section) are parsed to build
a STAC ItemCollection.

```python
from path import Path
from simplestac.utils import ItemCollection
from simplestac.local import collection_format, build_item_collection

col_file = tmpdir / "collection.json"

# Let's start from the example collection built as in static_stac.py
# directory containing the remote sensing scenes
image_dir = data_dir / "sentinel_data/dieback_detection_tutorial/study_area"
fmt = collection_format("S2_L2A_THEIA")
col = build_item_collection(image_dir, fmt)
col.save_object(col_file)

```

Anyone can make his own format depending on the naming of the file names.
The parsing is based on regex patterns.

The expected minimal structure is the following:
- item: metadata that are item-wise (i.e. independent of the asset)
  - pattern: how to parse the item id from the item directory
  - datetime: relative to datetime parsing (pattern, format)
- item_assets: the metadata relative to each asset
  - _asset key_:
    - pattern: regex pattern to find the band among the recursive list of files

See a simple [Theia format](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/simplestac/formats/S2_L2A_THEIA.json?ref_type=heads) made for the example.

## Extended pystac.ItemCollection

In `simplestac`, several methods have been added to the class `pystac.ItemCollection` in order to:
  - sort items,
  - filter (subset),
  - convert to xarray (with stackstac),
  - convert to geodataframe (with stac-geoparquet),
  - apply a function to each item or on a rolling window

After executing previous code, the following converts the ItemCollection into 
a geodataframe and plots its bouding box over the geometry of a region of interest :
```python
import geopandas as gpd

# Load the item collection
ItemCollection.from_file(col_file)

# Load the region of interest
roi = gpd.read_file(data_dir / "vector" / "area_interest.shp")

# Plot the collection geometry and the region of interest
ax = col.to_geodataframe().iloc[:1,:].to_crs(roi.crs).boundary.plot(color="red")
roi.plot(ax=ax)
```

See [examples](https://gitlab.irstea.fr/umr-tetis/stac/simplestac/-/tree/main/simplestac/examples)