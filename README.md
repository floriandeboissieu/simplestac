# SimpleSTAC : STAC tools to simplify STAC use.

[Documentation](https://simplestac-umr-tetis-stac-e5919c76d0463bc6d6669060b20af6f73f2a8.pages.mia.inra.fr)

[![licence](https://img.shields.io/badge/Licence-GPL--3-blue.svg)](https://www.r-project.org/Licenses/GPL-3)
[![python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org)

# Features

__Build a STAC ItemCollection based on local raster data:__

- functions to get the minimal raster metadata (bbox, geometry, projection) in STAC format
- class `MyStacItem` to create a simple STAC item with raster files
- function `build_item_collection` to build your small `ItemCollection` with a template for further metadata
  
__Extends class `pystac.ItemCollection` with methods to simplify data manipulation:__

- sort collection items
- filter (subset) cube by spatio-temporal coordinates and assets
- convert to a lazy dask `DataArray` cube
- convert to a geodataframe
- apply a function to each item or on a rolling window, write and add the created assets to current collection.

__Additional functions:__

- apply formula to a `DataArray` cube (e.g. "(B08-B04)/(B08+B04)")
- write collection assets to local files

# Get started

## Recommendations

Install the package in a virtual environment. See
[miniforge](https://github.com/conda-forge/miniforge) for conda/mamba, or 
[venv](https://docs.python.org/3/library/venv.html) for virtualenv.

## Install
Within a conda env:
```shell
mamba create -n simplestac --file https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/raw/main/environment.yml
```

Within a virtualenv:
```shell
pip install git+https://forgemia.inra.fr/umr-tetis/stac/simplestac
```

## Notebooks
Example notebooks make use of optional packages such as `ipykernel` or `xpystac`
which can be installed with:
```shell
# in conda env
mamba install -n simplestac --file https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/raw/main/environment-notebooks.yml
# or in venv
pip install git+https://forgemia.inra.fr/umr-tetis/stac/simplestac.git#egg=simplestac[notebook]
```

## Known issues

On Windows, the installation of `simplestac` may end with an error due to `Filename too long` in `stac-static`, a dependency of `simplestac`. Git can be configured to manage such filenames:
```shell
git config --system core.longpaths true
```

# Examples

See [example scripts](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples) for a detailed demo of `simpleSTAC` features.

Below is a small teaser.

## Example data
Example data can be downloaded [here](https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip) or with the script [download_data.py](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples/download_data.py?ref_type=heads)


## Create a local STAC ItemCollection

In this example, local files (see previous section) are parsed to build
a STAC ItemCollection.

```python
from path import Path
from simplestac.utils import ItemCollection
from simplestac.local import collection_format, build_item_collection

image_dir = data_dir / "s2_scenes"
res_dir = data_dir / "local_stac"
print(res_dir.mkdir_p())

col_file = res_dir / "collection.json"

# Let's start from the example collection built as in static_stac.py
# directory containing the remote sensing scenes
fmt = collection_format("S2_L2A_THEIA")
col = build_item_collection(image_dir, fmt)
col.save_object(col_file)
```

Anyone can make his own format depending on the naming of the file names.
The parsing is based on regex patterns.

The expected minimal structure is a json with the following:

- item: metadata that are item-wise (i.e. independent of the asset)
  - pattern: how to parse the item id from the item directory
  - datetime: relative to datetime parsing (pattern, format)
- item_assets: the metadata relative to each asset
  - _asset key_:
    - pattern: regex pattern to find the band among the recursive list of files

See a simple [Theia format](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/simplestac/formats/S2_L2A_THEIA.json?ref_type=heads) made for the example.

## Extended pystac.ItemCollection

After executing previous code, the following:

1. converts the created ItemCollection into a geodataframe and plots its bouding box over the geometry of a region of interest,
1. computes NDVI over a collection subset and plots it.

The same could be done with a remote item collection, see [examples](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples).

```python
import geopandas as gpd
from simplestac.utils import apply_formula
# Load the item collection
ItemCollection.from_file(col_file)

# Load the region of interest
roi = gpd.read_file(data_dir / "roi.geojson")

# Plot the collection geometry and the region of interest
ax = col.to_geodataframe().iloc[:1,:].to_crs(roi.crs).boundary.plot(color="red")
roi.plot(ax=ax)

# Compute the NDVI on a subset of the collection (clipped by datetime and geometry). Each NDVI raster is written in a local COG file and inserted in item assets in order to avoid memory overflow.
col.apply_items(
    fun=apply_formula, # a function that returns one or more xarray.DataArray
    name="NDVI",
    formula="((B08 - B04) / (B08 + B04))",
    output_dir=res_dir / "NDVI",
    datetime="2018-01-01/..",
    geometry=roi.geometry,
    inplace=True
)

# Select scenes with the NDVI
arr = col.filter(with_assets="NDVI").to_xarray()
assert "NDVI" in arr.band.values

# Plot NDVI applying cloud mask
mask = arr.sel(band="CLM") > 0
arr.sel(band="NDVI").where(~mask).isel(time=range(4)).plot(col="time", col_wrap=2)
```
