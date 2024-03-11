from simplestac.utils import write_assets, ItemCollection, harmonize_sen2cor_offet
import planetary_computer as pc
import pystac_client

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def test_to_xarray(roi, s2scene_pc_dir):
    time_range = "2022-01-01/2022-01-31"

    catalog = pystac_client.Client.open(URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        sortby="datetime",
    )
    col = ItemCollection(search.item_collection())
    x = col.drop_non_raster().to_xarray()
    assert len(x.time) == len(col)

def test_offset_harmonization(roi, s2scene_pc_dir):
    time_range = "2022-01-20/2022-01-31"

    catalog = pystac_client.Client.open(URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        sortby="datetime",
    )
    col = search.item_collection()
    harmonize_sen2cor_offet(col, inplace=True)
    of0 = col[0].assets["B02"].extra_fields["raster:bands"][0]["offset"]
    ofN = col[-1].assets["B02"].extra_fields["raster:bands"][0]["offset"]

    assert of0 == 0
    assert ofN == -1000

def test_drop_duplicates(roi, s2scene_pc_dir):
    time_range = "2022-01-20/2022-01-31"
    catalog = pystac_client.Client.open(URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        sortby="datetime",
    )
    col = search.item_collection()
    col1 = ItemCollection(col.clone()+col.clone())
    col1.drop_duplicates(inplace=True)
    assert len(col1) == len(col)

def test_write_assets(roi, s2scene_pc_dir):

    s2scene_pc_dir.rmtree_p().mkdir_p()
    time_range = "2016-01-01/2016-01-31"

    catalog = pystac_client.Client.open(URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=roi.to_crs(4326).total_bounds,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    col = ItemCollection(search.item_collection()).drop_non_raster()
    bbox = roi.to_crs(col.to_xarray().rio.crs).total_bounds
    col = pc.sign(col)
    encoding=dict(
        dtype="int16", 
        scale_factor=0.001,
        add_offset=0.0,
        _FillValue=-9999,
    )
    new_col = write_assets(col, s2scene_pc_dir, bbox=bbox, encoding=encoding)
    assert len(new_col) == len(col)
    assert len(new_col) == len(s2scene_pc_dir.dirs())
    item = new_col[0]
    assert item.id == col[0].id
    assert len(item.assets) == len(s2scene_pc_dir.dirs()[0].files("*.tif"))
    assert item.assets["B08"].href.startswith(s2scene_pc_dir)
    assert new_col[0].assets["B08"].extra_fields["raster:bands"][0]["scale"] == 0.001
    

    
