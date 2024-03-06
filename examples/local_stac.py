# %% [md]
# # Local STAC
# [Source code](https://forgemia.inra.fr/umr-tetis/stac/simplestac/examples/local_stac.py)
# 
# This notebook shows how to build a STAC ItemCollection from local data.
# %% [md]
# Load required libraries and define paths.
# %%
import geopandas as gpd
import numpy as np
from pandas import to_datetime
from path import Path
from simplestac.utils import ItemCollection, apply_formula
from simplestac.local import build_item_collection, collection_format
import xarray as xr
import rioxarray # necessary fo xr.open_dataset


data_dir = Path(__file__).parent/"data"
image_dir = data_dir / "s2_scenes"
roi_file = data_dir / "roi.geojson"
res_dir = data_dir / "local_stac"

if not (image_dir.exists() and roi_file.exists()):
    raise ValueError("Data not found")

# The notebook results will be stored in the
# following directory:
print(res_dir.mkdir_p())

# Define the path where the catalog will be written.
coll_path = res_dir / "collection.json"

# %% [md]
# Build a static collection with local files
# and save it in a json file:
# %%
if not coll_path.exists():
    coll = build_item_collection(image_dir, collection_format("S2_L2A_THEIA"))
    coll.save_object(coll_path)
    assert coll_path.exists()
    coll2 = ItemCollection.from_file(coll_path)
    assert set([item.id for item in coll2])==set([item.id for item in coll])

coll = ItemCollection.from_file(coll_path)
coll
# %% [md]
# Now let's browse the different methods added to ItemCollection.
# ## Convert to a geodataframe
# %%
coll.to_geodataframe()
# It can also include items in an additional `item` column
coll.to_geodataframe(include_items=True)
# By default the geodataframe is written in WGS84 (EPSG:4326)
# In order to have it in the items CRS (only if unique):
coll.to_geodataframe(wgs84=False).plot()

# %%[md]
# ## Sort items
# By date and tilename for example.
# This is particularly useful when applying a rolling window function on the data, see below.
coll.sort_items(by=["datetime", "s2:mgrs_tile"], inplace=True)

# %% [md]
# ## Filter items
# Subset collection by time, bounding box or any other property, e.g. S2 MGRS tile name
# %%
start = to_datetime("2016-01-01", utc=True)
end = to_datetime("2017-01-01", utc=True)
subcoll = coll.filter(datetime="2016-01-01/2017-01-01",
                      filter="s2:mgrs_tile = 'T31UFQ'")
assert set([item.datetime.timestamp() for item in subcoll])== \
    set([item.datetime.timestamp() for item in coll if \
         item.datetime>=start and item.datetime<=end])
subcoll
# %% [md]
# This operation could also be done after converting to xarray, see below.

# %% [md]
# ## Convert to xarray
# There are several ways to convert the collection to xarray.
# The simplest is to use method to_xarray() which returns a lazy Dask xarray.
# It is based on stackstac and provides a lot of options such as
# resolution, subset of assets, interpolion method, etc.
# See documentation for more details.
# %%
coll.to_xarray()
coll.to_xarray(resolution=60, assets=['B08', 'B8A'])
# %% [md]
# Here is are a few examples of converting collection to xarray while subsetting
# specific assets and a specific time period.
# %%
subxr = coll.filter(datetime="2016-01-01/2017-01-01", assets=['B08', 'B8A']).to_xarray()
subxr1 = coll.to_xarray(assets=['B08', 'B8A']).sel(time=slice('2016-01-01', '2017-01-01'))
subxr2 = coll.to_xarray().sel(band=['B08', 'B8A'], time=slice('2016-01-01', '2017-01-01'))
assert subxr.equals(subxr1)
assert subxr.equals(subxr2)

# %% [md]
# Filtering can also be done on the assets attributes.
# %%
subxr.sel(band = subxr.common_name=="red")

# %% [md]
# The ItemCollection can also be converted to xarray with xpystac plugin to xarray.
# In this last case band dimension was converted to variables, 
# which could also be done with `subxr.to_dataset(dim="band", promote_attrs=True)`.
# %%
subxr3 = xr.open_dataset(
    coll.filter(datetime="2016-01-01/2017-01-01", assets=['B08', 'B8A']),
    # don't forget to have xy_coords="center", 
    # otherwise the coordinates are wrong
    xy_coords="center", 
    )
subxr3 = subxr3.to_array("band").transpose("time", "band", "y", "x")
assert subxr.equals(subxr3)

# %% [md]
# Plot the first days.
# _Notice the interpolation (nearest) on the fly of B8A from 20m to 10m compared to B08 (10m)_
# %%
subxr[:2, :, :, :].plot(row='time', col='band') # 2 dates, 3 bands

# %% [md]
# Resolution can be changed on the fly
# when converting collection to xarray,
# see stackstac.stack for other option.
# %%
subxr4 = coll.filter(datetime="2016-01/2016-05").to_xarray(assets=['B08', 'B8A'], resolution=60)
subxr4.plot(row='time', col='band')


# %% [md]
# Extract a small zone.
# %%
subcoll_gdf = subcoll.to_geodataframe(wgs84=False)
extract_zone = subcoll_gdf.buffer(-1000).total_bounds
smallxr = subxr.rio.clip_box(*extract_zone)
smallxr
smallxr[:2,:,:,:].plot(row='time', col='band')

# or extract directly from to_xarray
# with a bbox
subcoll.to_xarray(bbox=extract_zone, assets=['B08', 'B8A'])
# or a geometry
roi = gpd.read_file(roi_file)
roi = roi.to_crs(subcoll.to_xarray().crs)
subcoll.to_xarray(
    geometry=roi.geometry, 
    assets=['B08', 'B8A']).isel(time=range(2)).plot(row='time', col='band')

# %% [md]
# ## Apply items
# The method `apply_items` applies to each item
# a function that returns one or more xarray.DataArray,
# writes the result in raster file, and
# includes it as a new asset in the item.
#
# The rasters are encoded by the way in int16,
# with a scale factor of 0.001, an offset of 0.0,
# and a nodata value of 2**15 - 1.
# For the NDVI for example:
# %%
coll.apply_items(
    fun=apply_formula, # a function that returns one or more xarray.DataArray
    name="NDVI",
    formula="((B08 - B04) / (B08 + B04))",
    output_dir=res_dir / "NDVI",
    datetime="2018-01-01/..",
    geometry=roi.geometry,
    inplace=True,
    writer_args=dict(
        encoding=dict(
            dtype="int16", 
            scale_factor=0.001,
            add_offset=0.0,
            _FillValue= 2**15 - 1,
        )
    )
)

coll.filter(with_assets="NDVI").to_xarray().sel(band="NDVI").isel(time=range(4)).plot(col="time", col_wrap=2)
# %%
coll.filter(with_assets="NDVI").to_xarray().sel(band="NDVI").isel(x=150, y=150).plot.line(x="time")

# %% [md]
# The method `apply_rolling` applies to a function to a group of items in
# a rolling window.
# %%
def masked_mean(x, band):
    if band not in x.band:
        return
    if x.time.size < 2:
        return # not enough data
    mask = x.sel(band="CLM")>0
    return x.sel(band=band).where(~mask).mean(dim="time", skipna=True)

coll.apply_rolling(
    fun=masked_mean,
    band="NDVI",
    name="mNDVI",
    output_dir=res_dir / "mNDVI",
    geometry=roi.geometry,
    inplace=True,
    window=5,
    center=True,
)
mask = coll.to_xarray().sel(band="CLM")>0
coll.filter(with_assets="mNDVI").to_xarray().where(~mask).sel(band=["mNDVI", "NDVI"]).isel(x=150, y=150).plot.line(x="time")

# %% [md]
# Et voilà!