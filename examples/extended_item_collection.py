# %%
# Load required libraries, create a temporary working directory
# and download the example dataset
from path import Path
from simplestac.utils import ItemCollection, apply_formula
import geopandas as gpd
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import zipfile
import json
import pystac_client
import planetary_computer as pc

# In the following, json.dumps is used to pretty print the nested dictionary
def pprint(d):
    print(json.dumps(d, sort_keys=False, indent=2))

tmpdir = Path(TemporaryDirectory(prefix="simplestac_").name)
tmpdir = Path("/tmp/simplestac_23yqa_87")
print(tmpdir) # to keep track of the directory to remove
data_dir = tmpdir/'fordead_data-main'

if not data_dir.exists():
    data_url = Path("https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip")

    with TemporaryDirectory() as tmpdir2:
        dl_dir = Path(tmpdir2)
        zip_path, _ = urlretrieve(data_url, dl_dir / data_url.name)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmpdir.mkdir_p())

# %%
### Load ROI and corresponding Planetary Computer S2 L2A collection
# Load the area of interest
roi = gpd.read_file(data_dir / "vector" / "area_interest.shp")

time_range = "2015-12-03/2019-09-20"

# Load the S2 L2A collection
# Here, the cloud cover is limited to 50% to limit the number of scenes
"https://planetarycomputer.microsoft.com/api/stac/v1"
url ="https://earth-search.aws.element84.com/v1"
catalog = pystac_client.Client.open(url)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=roi.to_crs(4326).total_bounds,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 20}},
)

# Make the search result an exetended ItemCollection, i.e. with exetended methods
col = ItemCollection(search.item_collection(), clone_items=False)
col.to_geodataframe().plot()


# Sign to have read access to all assets
# col = pc.sign(col)

# %%
# It is recommended to transform the geometry to the items CRS,
# otherwise it is transformed at each item processing.
epsg = col.to_geodataframe()["proj:epsg"].unique()[0]
geometry = roi.geometry.to_crs(epsg)

col.apply_items(apply_formula, 
                geometry=geometry,
                datetime="2016-01-01/2016-12-31",
                name="NDVI",
                output_dir=data_dir / "NDVI",
                formula="(nir-red)/(nir+red)", inplace=True)

col.to_xarray(assets=['NDVI']).plot(row='time')

col.filter(with_assets="NDVI").to_xarray(assets=['red', "green", "blue"], geometry=geometry).plot.imshow(rgb="band", row='time', robust=True)
# %%
import xarray as xr
xr.open_dataset(col.filter(datetime="2016-01-01/2016-12-31"))

# %%
import stackstac