# %% [md]
# # Remote STAC
# [Source code](https://forgemia.inra.fr/umr-tetis/stac/simplestac/-/blob/main/examples/remote_stac.py)
#
# This notebook shows how to mix remote and local data in a STAC ItemCollection.
# Two use case are shown, with Sentinel-2 L2A catalog Microsoft Planetary Computer
# and of Element84.

# %% [md]
# Load required libraries, define the paths of data and
# the region of interest (ROI), and load ROI.
# %%
import geopandas as gpd 
from path import Path
import pystac_client
import planetary_computer as pc
from simplestac.utils import ItemCollection, apply_formula, drop_assets_without_proj

data_dir = Path(__file__).parent/"data"
roi_file = data_dir / "roi.geojson"

if not roi_file.exists():
    raise ValueError("Data not found")

roi = gpd.read_file(roi_file)

# %% [md]
# ## Planetary Computer S2 L2A collection
# In the following, it:
# - searches for the time series of scenes in S2 L2A collection,
# limiting the scene cloud cover to 50%.
# - creates an ItemCollection
# - plots the collection geometry and the area of interest
# - saves the collection for future processing
# %%
URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
res_dir = (data_dir / "pc_stac").mkdir_p()

# time range
time_range = "2015-12-03/2019-09-20"

# file where the collection will be saved
col_path = res_dir / "collection.json"
if not col_path.exists():
    # Load the S2 L2A collection
    # Here, the cloud cover is limited to 50%
    # to limit the number of scenes
    catalog = pystac_client.Client.open(URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 50}},
    )

    # Make the search result an exetended ItemCollection, i.e. with exetended methods
    col = ItemCollection(search.item_collection(), clone_items=False)

    # Plot the collection geometry and the area of interest
    ax = col.to_geodataframe().iloc[:1,:].to_crs(roi.crs).boundary.plot(color="red")
    roi.plot(ax=ax)

    # Let's save the collection for furture
    col.save_object(col_path)

col = ItemCollection.from_file(col_path)
col

# %% [md]
# Drop non ratser assets may be needed
# to avoid issues in stackstac.stac
# with non-projected assets.
# %%
col.drop_non_raster(inplace=True)
col

# %% [md]
# ### Apply items
# The method `apply_items`, for each item in the collection:
# 1. applies a function which returns one
# or more `xarray.DataArray`,
# 2. saves the result in raster file,
# 3. includes the raster  as a new asset in the item.
# The code below shows the computation of the NDVI as an example.
# %%
# Sign to have read access to all assets.
col = pc.sign(col)
col.apply_items(
    fun=apply_formula, # a function that returns one or more xarray.DataArray
    name="NDVI",
    formula="((B08 - B04) / (B08 + B04))",
    output_dir=res_dir / "NDVI",
    datetime="2018-01-01/..",
    geometry=roi.geometry,
    inplace=True
)

# %%
col2 = col.filter(with_assets="NDVI", assets=["NDVI", "SCL"])
mask = ~col.to_xarray().sel(band="SCL").isin([4, 5])
arr = col2.to_xarray(geometry=roi.geometry).sel(band="NDVI")
# extract the local_stac notebook times
from pandas import to_datetime
times = ["2018-01-13", "2018-02-22", "2018-02-25", "2018-03-24"]
times = to_datetime(times)

# %% [md]
# Select a few scenes in the time series
# and plot the images.
# %%
subarr = arr.sel(time=arr.time.dt.floor("D").isin(times)).where(~mask)
subarr.plot(col="time", col_wrap=2)

# %% [md]
# Select a single pixel and plot its time series.
# %%
x = 642665
y = 5452555
col2.to_xarray().sel(band="NDVI").where(~mask).sel(x=x, y=y).plot.line(x="time")


# %% [md]
# ## Element84
# The same can be done with element84 catalog,
# although the STAC metadata is not exactly the same.
# Also, the data download can be slower than PC.
# %%
URL = "https://earth-search.aws.element84.com/v1"
res_dir = (data_dir / "e84_stac").mkdir_p() # results directory
# file where the collection will be saved
col_path = res_dir / "collection.json"

# time range
time_range = "2015-12-03/2019-09-20"

if not col_path.exists():
    # Load the S2 L2A collection
    # Here, the cloud cover is limited to 50% to limit the number of scenes
    catalog = pystac_client.Client.open(URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 50}},
    )

    # Make the search result an exetended ItemCollection, i.e. with exetended methods
    col = ItemCollection(search.item_collection(), clone_items=False)

    # Plot the collection geometry and the area of interest
    ax = col.to_geodataframe().iloc[:1,:].to_crs(roi.crs).boundary.plot(color="red")
    roi.plot(ax=ax)

    # Let's save the collection for furture
    col.save_object(col_path)

col = ItemCollection.from_file(col_path)
col

# %% [md]
# All assets are rasters in that collection,
# so no need to drop any asset.
#
# However, several things are to be noticed:
# - the assets are named with their color instead of band number
# - bands names are in lower case: SCL -> scl
# %%
col[0].assets.keys()

# %% [md]
# Thus, the NDVI formula will have to be adapted.
# 
# Moreover, notice that:
# - time has NaT value
# - several items are missing at the begining of 2018
# %%
col.to_xarray().time.values

# %% [md]
# The NaT value (Not a Time) is due to a time without nanoseconds,
# which is wrongly interpreted in stackstac.
# %%
col.filter(datetime="2018-06-28/2018-07-01")[0].datetime
col[0].datetime

# %% [md]
# Let's remove this item for the exercise.
# Another way around would have been to add a nanosecond to the time.
# %%
col.sort_items(by="datetime", inplace=True) # sort items by time
df = col.to_geodataframe(include_items=True) # convert to dataframe
good = (df.datetime<"2018-06-28")|(df.datetime>"2018-07-01")
col = ItemCollection(df.loc[good, :].item.to_list(), clone_items=False)
# check that nbo NaT is left
col.to_xarray().time.isnull()
# %% [md]
# ### Apply items
# Here, it computes the NDVI and saves the result in a raster file.
# %%
col.apply_items(
    fun=apply_formula, # a function that returns one or more xarray.DataArray
    name="NDVI",
    formula = "((nir - red) / (nir + red))",
    output_dir= res_dir / "NDVI",
    datetime="2018-01-01/..",
    geometry=roi.geometry,
    inplace=True
)
# The element84 download seems to take more time than PC.

# %%
times = ["2018-01-13", "2018-02-22", "2018-02-25", "2018-03-24"]
times = to_datetime(times)
col2 = col.filter(with_assets="NDVI", assets=["NDVI", "scl"])
mask = ~col.to_xarray().sel(band="scl").isin([4, 5])
arr = col2.to_xarray(geometry=roi.geometry).sel(band="NDVI")
# The times used in local_stac notebook are not available in Element84,
# due to the missing images in the collection as we said previously.
print(arr.time.dt.floor("D").isin(times))

# %% [md]
# Let's take other dates from `arr.time.values` and
# plot the images.
# %%
atimes = arr.sel(time=slice("2018-01-01", "2018-12-31")).time.values[:4]
arr.sel(time=arr.time.isin(atimes)).where(~mask).plot(col="time", col_wrap=2)

# %% [md]
# Select a point and plot its time series.
# %%
x = 642665
y = 5452555
col2.to_xarray().sel(band="NDVI").where(~mask).sel(x=x, y=y).plot.line(x="time")

# %% [md]
# Et voilà!
