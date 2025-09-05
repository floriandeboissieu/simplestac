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
def s2scene_zip_dir():
    scene_dir = here / "data" / "s2_scenes_zip"
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

@pytest.fixture(scope="session")
def pc_col2(roi):
    """
    This example is for unconsistent item properties:
    the property 's2:dark_features_percentage' was removed from
    N0510 to N0511.
    """
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    time_range = "2024-07-20/2024-08-11"
    catalog = pystac_client.Client.open(URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        sortby="datetime",
    )
    col = search.item_collection()
    yield col

@pytest.fixture(scope="session")
def maja_col(roi):
    URL = 'https://stacapi-cdos.apps.okd.crocc.meso.umontpellier.fr'
    collection = 'sentinel2-l2a-theia'
    time_range = "2016-01-01/2016-01-31"
    catalog = pystac_client.Client.open(URL)
    search = catalog.search(
        collections=[collection],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        sortby="datetime",
    )
    col = search.item_collection()
    yield col
