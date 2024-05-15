from simplestac.utils import write_assets, ItemCollection, harmonize_sen2cor_offset
import planetary_computer as pc
import pystac_client
from tempfile import TemporaryDirectory
import numpy as np

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def test_to_xarray(pc_col, roi):
    col = ItemCollection(pc_col)
    x = col.drop_non_raster().to_xarray()
    assert len(x.time) == len(col)

def test_offset_harmonization(pc_col):
    col = ItemCollection(pc_col)
    harmonize_sen2cor_offset(col, inplace=True)
    of0 = col[0].assets["B02"].extra_fields["raster:bands"][0]["offset"]
    ofN = col[-1].assets["B02"].extra_fields["raster:bands"][0]["offset"]

    assert of0 == 0
    assert ofN == -1000

def test_drop_duplicates(pc_col):
    col = ItemCollection(pc_col)
    col1 = ItemCollection(col.clone()+col.clone())
    assert len(col1) == 2*len(col)
    col1.drop_duplicates(inplace=True)
    assert len(col1) == len(col)

def test_drop_non_raster(pc_col):
    col = ItemCollection(pc_col)
    col1 = col.drop_non_raster()
    assert "preview" in col[0].assets
    assert "preview" not in col1[0].assets

def test_filter(pc_col):
    col = ItemCollection(pc_col)
    col1 = col.filter(assets="B02")
    assert "B03" in col[0].assets
    assert "B03" not in col1[0].assets

def test_write_assets(pc_col, roi, s2scene_pc_dir):

    s2scene_pc_dir.rmtree_p().mkdir_p()

    col = ItemCollection(pc_col)
    col.drop_non_raster(inplace=True)
    bbox = roi.to_crs(col.to_xarray().rio.crs).total_bounds
    encoding=dict(
        dtype="int16", 
        scale_factor=0.001,
        add_offset=0.0,
        _FillValue=-9999,
    )
    new_col = write_assets(col, s2scene_pc_dir, bbox=bbox, encoding=encoding, modifier=pc.sign_inplace)
    assert len(new_col) == len(col)
    assert len(new_col) == len(s2scene_pc_dir.dirs())
    item = new_col[0]
    assert item.id == col[0].id
    assert len(item.assets) == len(s2scene_pc_dir.dirs()[0].files("*.tif"))
    assert item.assets["B08"].href.startswith(s2scene_pc_dir)
    assert new_col[0].assets["B08"].extra_fields["raster:bands"][0]["scale"] == 0.001

    with TemporaryDirectory(prefix="simplestac-tests_") as tempdir:
        new_col2 = write_assets(col, tempdir, geometry=roi.buffer(5), encoding=encoding, modifier=pc.sign_inplace)
        assert len(new_col2) == len(new_col)
        assert new_col2[0].bbox == new_col[0].bbox



    
