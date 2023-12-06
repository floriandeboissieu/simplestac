"""This notebook aims at showing how to build a STAC catalog from local data"""

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
# sort the collectionb, by date and tilename for example
coll2.sort_items(by=["datetime", "s2:mgrs_tile"], inplace=True)

# %%
# convert collection to a lazy Dask xarray
coll2.to_xarray()
# several options are available (resolution, subset of assets, choice of the interpolion method, etc.),
# see stackstac.stack for more details. Example:
coll2.to_xarray(resolution=60)

# %%
# subset collection by time, bounding box or any other property, e.g. S2 MGRS tile name
start = to_datetime("2016-01-01")
end = to_datetime("2017-01-01")
subcoll = coll.filter(datetime="2016-01-01/2017-01-01",
                      filter="s2:mgrs_tile = 'T31UFQ'")
assert set([item.datetime.timestamp() for item in subcoll])== \
    set([item.datetime.timestamp() for item in coll if \
         item.datetime>=start and item.datetime<=end])
subcoll

# %%
# Here is an example of converting collection to xarray while subsetting
# specific assets and a specific time period.
# It can be done in several ways, at the collection level or at the xarray level,
# the first method being slightly faster.
subxr = coll.filter(datetime="2016-01-01/2017-01-01").to_xarray(assets=['B08', 'B8A'])
subxr2 = coll.filter(datetime="2016-01-01/2017-01-01").to_xarray().sel(band=['B08', 'B8A'])
assert subxr.equals(subxr2)
subxr.sel(band = subxr.common_name=="red")

# %%
# The ItemCollection can also be converted to xarray with xpystac plugin to xarray.
# In this last case band dimension was converted to variables, 
# which could also be done with subxr.to_dataset(dim="band", promote_attrs=True)
subxr3 = xr.open_dataset(coll.filter(datetime="2016-01-01/2017-01-01",
filter="s2:mgrs_tile='T31UFQ'"))[['B08', 'B8A']]
subxr3

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
# convert collection to geodataframe
# **warning**: geometry CRS is WGS84 (EPSG:4326)
subcoll_gdf = subcoll.to_geodataframe(include_items=True)
coll_crs = subcoll_gdf.loc[0,'proj:epsg']
subcoll_gdf.to_crs(coll_crs, inplace=True)
subcoll_gdf

# %%
# extract a small zone
extract_zone = subcoll_gdf.buffer(-1000).total_bounds
smallxr = subxr.rio.clip_box(*extract_zone)
smallxr
smallxr[:2,:,:,:].plot(row='time', col='band')

# %%
# remove temp directory **recursively**
tmpdir.rmtree()

# Et voilà!








# %%
