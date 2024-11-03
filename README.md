# SimpleSTAC : STAC tools to simplify STAC use.

[![version](https://img.shields.io/gitlab/v/tag/10090?gitlab_url=https%3A%2F%2Fforgemia.inra.fr&label=version&color=green)](https://forgemia.inra.fr/umr-tetis/stac/simplestac)
[![licence](https://img.shields.io/badge/Licence-GPL--3-blue.svg)](https://www.r-project.org/Licenses/GPL-3)
[![python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org)
[![build status](https://forgemia.inra.fr/umr-tetis/stac/simplestac/badges/main/pipeline.svg)](https://forgemia.inra.fr/umr-tetis/stac/simplestac/pipelines/main/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13738413.svg)](https://doi.org/10.5281/zenodo.13738413)

__[Documentation](https://umr-tetis.pages.mia.inra.fr/stac/simplestac)__

STAC, i.e. Spatio-Temporal Asset Catalog, is a standard especially useful to present a spatio-temporal catalog of raster files,
typically time series of satellite data:
- when making a request to a remote STAC server, an ItemCollection is returned, i.e. a list of items. 
  This ItemCollection can be saved to a json file, e.g. to keep track of all the rasters used in a processing.
- if raster files are formated with Cloud Optimized GeoTIFF, it allows a fast extraction of sparse pixels
  (e.g. polygons or points) from a large remote time series.

This package aims at simplifying the way to process such an ItemCollection,
mixing remote and local files for example, 
and to build your own ItemCollection of your local raster files.

This way the same processing line can be easily scaled up from a small experiment on local files
to a production context at larger scale, preparing the results for publication in a new STAC collection.

# Features

__Build a STAC ItemCollection based on local raster files:__

- convert a series of raster files to STAC assets
- create a STAC item from raster directory, e.g. Sentinel-2 scene, see class `MyStacItem` 
- function `build_item_collection` to build your small `ItemCollection` with a template for further metadata
  
__Extends class `pystac.ItemCollection` with methods to simplify its manipulation:__

- convert ItemCollection to a geodataframe with image footprint geometry
- convert ItemCollection to a lazy dask `DataArray` cube
- fast extract sparse points of the ItemCollection (e.g. 90s for 700 points full S2 time series with all bands)
- subset ItemCollection a-posteriori, e.g. filter according to STAC attributes, crop according to geometry
- apply a function to each item of the collection, saving new assets to files and adding them to the processed ItemCollection
- sort ItemCollection
- harmonize items scale and offset (e.g. for different Sentinel-2 processing baselines)
- drop non raster assets


__Additional functions:__

- apply a band formula to a `DataArray` cube (e.g. "(B08-B04)/(B08+B04)")
- write an ItemCollection to a series of local files, see `write_assets`


# Install

We recommend installing the package in a virtual environment. See
[miniforge](https://github.com/conda-forge/miniforge) for conda/mamba, or 
[venv](https://docs.python.org/3/library/venv.html) for virtualenv.


Create a conda env:
```shell
mamba env create -n simplestac --file https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/raw/main/environment.yml
```

Within a virtualenv:
```shell
pip install git+https://forgemia.inra.fr/umr-tetis/stac/simplestac
```

Update `simplestac` within an env (conda or virtualenv):
```shell
pip install git+https://forgemia.inra.fr/umr-tetis/stac/simplestac
```

__Known issues:__

The installation of `simplestac` on Windows may end up with an error due to `Filename too long` at `stac-static` installation,
a dependency of `simplestac`. Git can be configured to manage such filenames:
```shell
git config --system core.longpaths true
```

# Examples

Below is a small teaser.

For a detailed demo of `simpleSTAC` features, see [example scripts](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples).

Example data can be downloaded [here](https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip) or with the script [download_data.py](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples/download_data.py?ref_type=heads)

## Extract a study area from a remote collection

In this example:
1. a region of interest is extracted from
the Planetary Computer Sentinel-2 L2A collection,
1. extraction is written down to a local files
1. a STAC ItemCollection is returned and saved

```python
import geopandas as gpd 
from path import Path
import pystac_client
import planetary_computer as pc
from simplestac.utils import ItemCollection, apply_formula, drop_assets_without_proj

data_dir = Path(__file__).parent/"data"
roi_file = data_dir / "roi.geojson"

# load region of interest
roi = gpd.read_file(roi_file)

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
res_dir = (data_dir / "pc_stac").mkdir_p()

# time range
time_range = "2016-01-01/2020-01-01"

# file where the collection will be saved
col_path = res_dir / "collection.json"

# Load the S2 L2A collection limiting cloud cover to 50%
catalog = pystac_client.Client.open(URL)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=roi.to_crs(4326).total_bounds,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 50}},
)

# Make the search result an exetended ItemCollection, i.e. with exetended methods
col = ItemCollection(search.item_collection(), clone_items=False)
# Drop non raster assets
col.drop_non_raster(inplace=True)

# write down cropped item collection
col = write_assets(col, output_dir=res_dir/"sentinel-2-l2a", geometry=roi)
```

## Create a local STAC ItemCollection

In this example, a series of Sentinel-2 L2A scenes stored locally are parsed to build
a local STAC ItemCollection.

The file tree looks like that (pruned to keep it readable):
```shell
s2_scenes
├── SENTINEL2A_20170526-105518-082_L2A_T31UFQ_D_V1-4
├── SENTINEL2A_20171202-105415-464_L2A_T31UFQ_C_V2-2
├── SENTINEL2A_20190725-105725-224_L2A_T31UFQ_C_V2-2
├── SENTINEL2B_20170707-104022-457_L2A_T31UFQ_C_V2-2
└── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2
    ├── MASKS
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B11.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B12.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B2.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B3.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B4.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B5.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B6.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B7.tif
    ├── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B8A.tif
    └── SENTINEL2B_20170806-104021-455_L2A_T31UFQ_C_V2-2_FRE_B8.tif
```

The STAC ItemCollection can be build and saved to a json file with:
```python
from path import Path
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

## Process the ItemCollection

After executing previous code, the following:

1. plot a region of interest over a the ItemCollection footprint,
1. computes NDVI over a collection subset and plots it.

The same could be done with a remote item collection, see [examples](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples).

```python
import geopandas as gpd
from simplestac.utils import ItemCollection, apply_formula
# Load previously saved item collection
col = ItemCollection.from_file(col_file)

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
