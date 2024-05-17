import geopandas as gpd
import logging
from path import Path
import planetary_computer as pc
import pystac_client
import pytest

logging.basicConfig(level=logging.INFO)

here = Path(__file__).parent
download_script = here / "download_data.py"
print("Downloading the test data...")
exec(open(download_script).read())

@pytest.fixture(scope="session")
def s2scene_dir():
    scene_dir = here / "data" / "s2_scenes"
    yield scene_dir

@pytest.fixture(scope="session")
def roi_file():
    roi_file = here / "data" / "roi_small.geojson"
    yield roi_file

@pytest.fixture(scope="session")
def roi(roi_file):
    roi = gpd.read_file(roi_file)
    yield roi

@pytest.fixture(scope="session")
def s2scene_pc_dir():
    scene_dir = here / "data" / "s2_scenes_pc"
    yield scene_dir

@pytest.fixture(scope="session")
def pc_col(roi):
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    time_range = "2022-01-20/2022-01-31"
    catalog = pystac_client.Client.open(URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        sortby="datetime",
    )
    col = search.item_collection()
    yield col