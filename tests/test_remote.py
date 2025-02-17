from simplestac.utils import write_assets, ItemCollection, harmonize_sen2cor_offset
import planetary_computer as pc
import pystac_client
from tempfile import TemporaryDirectory
import numpy as np


def test_projv2_to_projv12(maja_col):
    col = ItemCollection(maja_col)
    ef = col[0].assets["B02"].extra_fields
    assert "proj:epsg" in ef
    assert isinstance(ef["proj:epsg"], int)
    col[0].validate()

def test_filter_assets(pc_col):
    col = ItemCollection(pc_col)
    col1 = col.filter_assets(assets=["B02", "B03"])
    assert len(col1[0].assets) == 2
    col1 = col.filter_assets(assets=["B02"], drop=True)
    assert "B02" not in col1[0].assets
    col1 = col.filter_assets(pattern="^proj:bbox", drop=False)
    assert all(["proj:bbox" in a.extra_fields for a in col1[0].assets.values()])

def test_to_xarray(pc_col):
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

def test_update_scale_offset(pc_col):
    from simplestac.utils import update_scale_offset
    col = ItemCollection(pc_col)
    scale = 1e-4
    offset = -0.1
    col1 = update_scale_offset(col, scale, offset)
    assert col1[0].assets["B02"].extra_fields["raster:bands"][0]["scale"] == 0.0001
    assert col1[0].assets["B02"].extra_fields["raster:bands"][0]["offset"] == -0.1

    col2 = harmonize_sen2cor_offset(col)
    col1 = update_scale_offset(col2, scale)
    assert col1[0].assets["B02"].extra_fields["raster:bands"][0]["scale"] == 0.0001
    assert col1[0].assets["B02"].extra_fields["raster:bands"][0]["offset"] == 0
    assert col1[-1].assets["B02"].extra_fields["raster:bands"][0]["scale"] == 0.0001
    assert col1[-1].assets["B02"].extra_fields["raster:bands"][0]["offset"] == -0.1

    v = col.drop_non_raster().to_xarray().isel(time=-1, x=0, y=0).sel(band="B02").values
    v1 = col1.drop_non_raster().to_xarray().isel(time=-1, x=0, y=0).sel(band="B02").values
    assert not np.isnan(v)
    assert v*scale+offset == v1

    col1 = update_scale_offset(col2, scale)
    col2 = harmonize_sen2cor_offset(col)
    v = col.drop_non_raster().to_xarray().isel(time=-1, x=0, y=0).sel(band="B02").values
    v1 = col1.drop_non_raster().to_xarray().isel(time=-1, x=0, y=0).sel(band="B02").values
    assert v*scale+offset == v1

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

def test_to_geodataframe(pc_col2):
    col = ItemCollection(pc_col2)
    col.to_geodataframe()

    
