"""This notebook aims at showing how to build a STAC ItemCollection from local data"""

# %%
# load required libraries, create a temporary working directory
# and download the example dataset
from pandas import to_datetime
from path import Path
from simplestac.utils import ItemCollection
from simplestac.local import build_item_collection, collection_format
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import xarray as xr
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

image_dir = data_dir / "sentinel_data/dieback_detection_tutorial/study_area"

# make a temporary directory where the catalog will be written
coll_path = tmpdir / "collection.json"

# %%
# build a static collection with local files
coll = build_item_collection(image_dir, collection_format("S2_L2A_THEIA"))
coll

# %%
# save collection in json format
coll.save_object(coll_path)
assert coll_path.exists()

# %%
# load collection
coll2 = ItemCollection.from_file(coll_path)
assert set([item.id for item in coll2.items])==set([item.id for item in coll.items])


# %%
# Now let's browse the different methods added to ItemCollection
# ## Convert to a geodataframe
coll.to_geodataframe()
# It can also include items in an additional `item` column
coll.to_geodataframe(include_items=True)
# By default the geodataframe is written in WGS84 (EPSG:4326)
# In order to have it in the items CRS (only if unique):
coll.to_geodataframe(wgs84=False).plot()

# %%
# ## Sort items
# By date and tilename for example
coll2.sort_items(by=["datetime", "s2:mgrs_tile"], inplace=True)
# This is particularly useful when applying a rolling window function on the data, see below.

# %%
# ## Filter items
# subset collection by time, bounding box or any other property, e.g. S2 MGRS tile name
start = to_datetime("2016-01-01")
end = to_datetime("2017-01-01")
subcoll = coll.filter(datetime="2016-01-01/2017-01-01",
                      filter="s2:mgrs_tile = 'T31UFQ'")
assert set([item.datetime.timestamp() for item in subcoll])== \
    set([item.datetime.timestamp() for item in coll if \
         item.datetime>=start and item.datetime<=end])
subcoll

# This operation could also be done after converting to xarray, see below.

# %%
# ## Convert to xarray
# There are several ways to convert the collection to xarray.
# The simplest is to use method to_xarray() which returns a lazy Dask xarray.
# It is based on stackstac and provides a lot of options such as
# resolution, subset of assets, interpolion method, etc.
# See documentation for more details.
coll2.to_xarray()
coll2.to_xarray(resolution=60, assets=['B08', 'B8A'])

# Here is are a few examples of converting collection to xarray while subsetting
# specific assets and a specific time period.
subxr = coll.filter(datetime="2016-01-01/2017-01-01", assets=['B08', 'B8A']).to_xarray()
subxr1 = coll.to_xarray(assets=['B08', 'B8A']).sel(time=slice('2016-01-01', '2017-01-01'))
subxr2 = coll.to_xarray().sel(band=['B08', 'B8A'], time=slice('2016-01-01', '2017-01-01'))
assert subxr.equals(subxr1)
assert subxr.equals(subxr2)

# filtering can also be done on the assets attributes
subxr.sel(band = subxr.common_name=="red")


# The ItemCollection can also be converted to xarray with xpystac plugin to xarray.
# In this last case band dimension was converted to variables, 
# which could also be done with subxr.to_dataset(dim="band", promote_attrs=True)
subxr3 = xr.open_dataset(
    coll.filter(datetime="2016-01-01/2017-01-01", assets=['B08', 'B8A']),
    # don't forget to have xy_coords="center", 
    # otherwise the coordinates are wrong
    xy_coords="center", 
    )
subxr3 = subxr3.to_array("band").transpose("time", "band", "y", "x")
assert subxr.equals(subxr3)

# %%
# plot the first days
# notice the interpolation (nearest) on the fly of B8A from 20m to 10m compared to B08 (10m)
subxr[:2, :, :, :].plot(row='time', col='band') # 2 dates, 3 bands

# %%
# resolution can be changed on the fly
# when converting collection to xarray
# see stackstac.stack for other option
subxr4 = coll.filter(datetime="2016-01/2016-05").to_xarray(assets=['B08', 'B8A'], resolution=60)
subxr4.plot(row='time', col='band')


# %%
# move up to 
# extract a small zone
subcoll_gdf = subcoll.to_geodataframe(wgs84=False)
extract_zone = subcoll_gdf.buffer(-1000).total_bounds
smallxr = subxr.rio.clip_box(*extract_zone)
smallxr
smallxr[:2,:,:,:].plot(row='time', col='band')

# or extract directly from to_xarray
# with a bbox
subcoll.to_xarray(bbox=extract_zone, assets=['B08', 'B8A'])
# or a geometry
import geopandas as gpd
roi = gpd.read_file(data_dir / "vector" / "area_interest.shp")
roi = roi.to_crs(subcoll.to_xarray().crs)
subcoll.to_xarray(
    geometry=roi.geometry, 
    assets=['B08', 'B8A']).isel(time=range(2)).plot(row='time', col='band')
# %%
# ## Apply items
# The method `apply_items`` applies to each item
# a function that returns one or more xarray.DataArray,
# saves the result in raster file, and
# includes it as a new asset in the item.
# For the NDVI for example:
from simplestac.utils import apply_formula
coll2.apply_items(
    fun=apply_formula, # a function that returns one or more xarray.DataArray
    name="NDVI",
    formula = "((B08 - B04) / (B08 + B04))",
    output_dir=tmpdir / "NDVI",
    datetime="2018-01-01/..",
    geometry=roi.geometry,
    inplace=True
)
coll2.filter(with_assets="NDVI").to_xarray().sel(band="NDVI").isel(time=range(4)).plot(col="time", col_wrap=2)
# %%
coll2.filter(with_assets="NDVI").to_xarray().sel(band="NDVI").isel(x=150, y=150).plot.line(x="time")
# %%
# The method `apply_rolling` applies to a function to a group of items in
# a rolling window.
import numpy as np
def masked_mean(x, band):
    if band not in x.band:
        return
    if x.time.size < 2:
        return # not enough data
    mask = x.sel(band="CLM")>0
    return x.sel(band=band).where(~mask).mean(dim="time", skipna=True)

coll2.apply_rolling(
    fun=masked_mean,
    band="NDVI",
    name="mNDVI",
    output_dir=tmpdir / "mNDVI",
    geometry=roi.geometry,
    inplace=True,
    overwrite=True,
    window=5,
    center=True,
)
mask = coll2.to_xarray().sel(band="CLM")>0
coll2.filter(with_assets="mNDVI").to_xarray().where(~mask).sel(band=["mNDVI", "NDVI"]).isel(x=150, y=150).plot.line(x="time")


# %%
# remove temp directory **recursively**
tmpdir.rmtree()

# Et voilà!








# %%
