from simplestac.local import collection_format, build_item_collection
from simplestac.utils import write_raster, apply_formula
import xarray as xr
import pystac
from shapely.geometry import MultiPoint
import geopandas as gpd


def test_formatting():
    fmt = collection_format()
    assert fmt["item"]["pattern"] == '(SENTINEL2[AB]_[0-9]{8}-[0-9]{6}-[0-9]{3}_L2A_T[0-9A-Z]{5}_[A-Z]_V[0-9]-[0-9])'

def test_build(s2scene_dir):
    col = build_item_collection(s2scene_dir, collection_format())
    assert len(col) == len(s2scene_dir.dirs())
    assert len(col[0].assets) == 11

def test_datetime(s2scene_dir):
    fmt = collection_format()
    item = fmt["item"]
    prop = item["properties"]
    dt = prop.pop("datetime")
    # case datetime is in item
    item["datetime"] = dt
    col = build_item_collection(s2scene_dir, fmt)
    # case datetime is replaced by start_datetime and end_datetime
    item["start_datetime"] = item["end_datetime"] = dt
    col = build_item_collection(s2scene_dir, fmt)
    assert len(col) == len(s2scene_dir.dirs())
    # case datetime is in properties
    item.pop("start_datetime")
    item.pop("end_datetime")
    prop["datetime"] = dt
    col = build_item_collection(s2scene_dir, fmt)
    assert len(col) == len(s2scene_dir.dirs())
    # case start_datetime and end_datetime are in properties
    prop.pop("datetime")
    prop["start_datetime"] = prop["end_datetime"] = dt
    col = build_item_collection(s2scene_dir, fmt)
    assert len(col) == len(s2scene_dir.dirs())

def test_apply_items(s2scene_dir, roi):
    col = build_item_collection(s2scene_dir, collection_format())
    NDVI_dir = s2scene_dir.parent / "NDVI"
    NDVI_dir.rmtree_p()
    col.apply_items(
        apply_formula, 
        name="NDVI",
        geometry=roi.geometry,
        formula="((B08-B04)/(B08+B04))",
        output_dir=NDVI_dir,
        inplace=True)
    assert "NDVI" in col.items[-1].assets
    assert len(NDVI_dir.files()) == len(col)
    # check if COG
    assert col.items[-1].assets["NDVI"].media_type == pystac.MediaType.COG

def test_apply_rolling(s2scene_dir):
    col = build_item_collection(s2scene_dir, collection_format())
    output_dir = s2scene_dir.parent / "B07_diff"
    output_dir.rmtree_p()
    def B07_diff(x):
        if len(x.time) > 1:
            return x.sel(band="B07").diff("time")
    col.sort_items(by="datetime", inplace=True) 
    col.apply_rolling(
        B07_diff, 
        name="B07_diff",
        output_dir=output_dir,
        inplace=True,
        window=2)
    assert "B07_diff" in col.items[-1].assets
    assert len(output_dir.files()) == (len(col)-1)

  

def test_apply_items_raster_args(s2scene_dir, roi):
    col = build_item_collection(s2scene_dir, collection_format())
    output_dir = s2scene_dir.parent / "NDVI"
    output_dir.rmtree_p()
    col1 = col.apply_items(
        apply_formula,
        name="NDVI",
        formula="((B08-B04)/(B08+B04))",
        output_dir=output_dir,
        geometry=roi.geometry,
        writer_args=dict(
            encoding=dict(
                dtype="int16", 
                scale_factor=0.001,
                add_offset=0.0,
                _FillValue= -2**15,
            ),
        )
    )
    
    rb = col1[0].assets["NDVI"].extra_fields["raster:bands"][0]
    assert rb["datatype"] == "int16"
    assert rb["scale"] == 0.001
    assert rb["offset"] == 0.0
    assert rb["nodata"] == -2**15

def test_extract_points(s2scene_dir, roi):
    col = build_item_collection(s2scene_dir, collection_format())
    points = roi.geometry.apply(lambda x: MultiPoint(list(x.exterior.coords)))
    points.index.rename("id_point", inplace=True)
    ext = col.extract_points(points)
    assert ext.id_point.isin(points.index.values).all()
    coords = points.get_coordinates().reset_index(drop=True)
    points = gpd.GeoSeries(gpd.points_from_xy(**coords, crs=roi.crs))
    points.index.rename("id_point", inplace=True)
    ext = col.extract_points(points)
    assert ext.id_point.isin(points.index.values).all()

############################################################
    


    