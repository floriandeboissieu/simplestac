import pystac
import pytest
from path import Path
import geopandas as gpd

here = Path(__file__).parent
download_script = here / "download_data.py"
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