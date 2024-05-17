"""
This module aims at providing a simple interface to build a STAC ItemCollection
from a series of scenes/images in local files.
"""
## TODO: see if rio-cogeo would simplify the code and externalise
# the parsing of the metadata from the band files.

from datetime import timedelta
import json
import geopandas as gpd
import logging
from path import Path
import rasterio
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio import warp
from rasterio.features import bounds as feature_bounds
import re
import pandas as pd
from pandas import DataFrame, to_datetime
import pystac
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pystac.extensions import eo, projection, raster
from shapely.geometry import box
from shapely import to_geojson
from shapely.ops import unary_union
from tqdm import tqdm
import warnings

from rio_cogeo.cogeo import cog_validate

EPSG_4326 = rasterio.crs.CRS.from_epsg(4326)

logger = logging.getLogger(__name__)

############# Generic functions to build collection
def get_rio_info(file):
    """
    Get information from a raster file.

    Parameters
    ----------
    file : str
        The path or uri to the raster file.

    Returns
    -------
    tuple
        A tuple containing the bounding box of the raster, the media type,
        the ground sample distance (gsd), and the metadata.
    """
    # file = "~/git/fordeadv2/fordead_data-main/sentinel_data/dieback_detection_tutorial/study_area/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1_FRE_B11.tif"
    file = Path(file).expand()
    with rasterio.open(file) as src:
        bbox = src.bounds
        meta = src.meta
        media_type = get_media_type(src)
        gsd = src.res[0]
        tags = src.tags()
        scales = src.scales
        offsets = src.offsets
    
    # If needed at some point, we could test MediaType.COG with rio_cogeo.cogeo.cog_validate  
    if media_type == pystac.MediaType.GEOTIFF:
        iscog, _, _ = cog_validate(file)
        if iscog:
            media_type = pystac.MediaType.COG
        
    return bbox, media_type, gsd, meta, tags, scales, offsets

def get_media_type(
    src_dst: Union[DatasetReader, DatasetWriter, WarpedVRT, MemoryFile]
) -> Optional[pystac.MediaType]:
    """Find MediaType for a raster dataset.
    Copied from rio-stac (https://github.com/developmentseed/rio-stac)
    """
    driver = src_dst.driver

    if driver == "GTiff":
        if src_dst.crs:
            return pystac.MediaType.GEOTIFF
        else:
            return pystac.MediaType.TIFF

    elif driver in [
        "JP2ECW",
        "JP2KAK",
        "JP2LURA",
        "JP2MrSID",
        "JP2OpenJPEG",
        "JPEG2000",
    ]:
        return pystac.MediaType.JPEG2000

    elif driver in ["HDF4", "HDF4Image"]:
        return pystac.MediaType.HDF

    elif driver in ["HDF5", "HDF5Image"]:
        return pystac.MediaType.HDF5

    elif driver == "JPEG":
        return pystac.MediaType.JPEG

    elif driver == "PNG":
        return pystac.MediaType.PNG

    warnings.warn("Could not determine the media type from GDAL driver.")
    return None

def stac_proj_info(bbox, gsd, meta):
    """Projection information returned in the STAC format.

    It converts typical 

    Parameters
    ----------
    bbox : Bounds or list object
        Bounding box, e.g. rasterio `bounds` attribute.
    gsd : float
        Ground sample distance, e.g. rasterio `res[0]` attribute.
    meta : dict
        Metadata dict returned by rasterio.

    Returns
    -------
    dict
        epsg, wkt2, geometry, centroid, bbox, shape, transform
        with prefix `proj:`, and gsd
    """
    # most could be done by get_projection_info from
    # https://github.com/developmentseed/rio-stac/blob/main/rio_stac/stac.py
    # centroid nor gsd would be available...

    epsg = meta["crs"].to_epsg()
    wkt2 = meta["crs"].to_wkt()
    # _, _, centroid = bbox_to_wgs(bbox, meta["crs"])
    proj = dict(
        epsg = epsg,
        wkt2 = wkt2,
        # geometry = bbox_to_geom(bbox),
        geometry = bbox_to_geom(bbox),
        # centroid = centroid,
        bbox = list(bbox),
        shape = (meta["height"], meta["width"]),
        transform = list(meta["transform"])
    )

    proj_info = {
            f"proj:{name}": value
            for name, value in proj.items()
    }
    proj_info.update(gsd = gsd)
    
    return proj_info

def stac_raster_info(meta, tags, gsd, scales, offsets):
    """Raster information returned in the STAC format.

    Parameters
    ----------
    meta : dict
        Metadata dict returned by rasterio.
    tags : dict
        Tags returned by rasterio.
    gsd : float
        Ground sample distance.
    scales : list
        Scales returned by rasterio.
    offsets : list
        Offsets returned by rasterio.

    Returns
    -------
    dict
        STAC extension raster information, with prefix `raster:bands`
    
    See also
    --------
    simplestac.local.get_rio_info
    
    Notes
    -----
    See https://github.com/stac-extensions/raster
    """
    bands = [{}]
    if "nodata" in meta and meta["nodata"] is not None:
        bands[0]["nodata"] = meta["nodata"]
    if "AREA_OR_POINT" in tags:
        bands[0]["sampling"] = tags["AREA_OR_POINT"].lower()
    if "dtype" in meta:
        bands[0]["datatype"] = meta["dtype"]
    
    # 'resolution' is not always in tags, thus gsd is used instead.
    bands[0]["spatial_resolution"] = gsd
    
    if scales is not None:
        bands[0]["scale"] = scales[0]
    if offsets is not None:
        bands[0]["offset"] = offsets[0]
    
    return {"raster:bands": bands}


def bbox_to_geom(bbox):
    """
    Convert a bounding box to a geometry object.

    Parameters
    ----------
    bbox : tuple or list
        The bounding box represented as a tuple of four values (xmin, ymin, xmax, ymax).

    Returns
    -------
    dict
        The geometry object represented as a dictionary.

    Notes
    -----
    This function is 3 times faster than shapely.geometry.mapping.

    Examples
    --------
    >>> bbox_to_geom((0, 0, 10, 10))
    {'type': 'Polygon', 'coordinates': (((0, 0), (10, 0), (10, 10), (0, 10), (0, 0)),)}
    """
    # 3x faster than shapely.geometry.mapping
    return json.loads(to_geojson(box(*bbox)))

def bbox_to_wgs(bbox, epsg):
    """
    Convert a bounding box to WGS84 coordinates.

    Parameters
    ----------
    bbox : tuple or list
        A tuple representing the bounding box coordinates (xmin, ymin, xmax, ymax).
    epsg : int
        The EPSG code representing the coordinate reference system.

    Returns
    -------
    bbox : tuple
        A tuple representing the updated bounding box coordinates.
    geom : dict
        A dictionary representing the transformed geometry.
    """
    geom = bbox_to_geom(bbox)
    # Reproject the geometry to "epsg:4326"
    epsg = rasterio.crs.CRS.from_epsg(epsg)
    geom = warp.transform_geom(epsg, EPSG_4326, geom)
    bbox = feature_bounds(geom)

    return bbox, geom
#######################################

############# Build collection from json format
FORMATS_DIR = Path(__file__).parent / "formats"

def common_name_table():
    """
    Returns a table of common band names and their corresponding
    wavelenght minimum and maximum values.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame containing the common band names and their corresponding wavelength minimum and maximum values.
    """
    # https://github.com/stac-extensions/eo/#common-band-names
    file = FORMATS_DIR / "common_name_table.csv"
    table = pd.read_csv(file, sep="\t")
    table["band_min"] = table.iloc[:,1].apply(lambda x: float(x.split("-")[0]))
    table["band_max"] = table.iloc[:,1].apply(lambda x: float(x.split("-")[1]))
    return table

# CN_TABLE = common_name_table()
#### Not working, e.g. band 5 wvl-fwhm/2 of Sentinel 2 is out range rededge...
# def band_to_common_name(centre_wavelength, full_width_half_max):
#     return CN_TABLE.loc[((centre_wavelength-full_width_half_max/2)>=CN_TABLE.band_min) & ((centre_wavelength+full_width_half_max/2) <= CN_TABLE.band_max), "Common Name"].values

def collection_format(type="S2_L2A_THEIA", formats_dir=FORMATS_DIR):
    """
    Loads the collection format for a specified type.

    The format is loaded from JSON file with path `{formats_dir}/{type}.json`.

    Parameters
    ----------
    type : str, optional
        The type of collection format to load. Defaults to "S2_L2A_THEIA".
    formats_dir : str, optional
        The directory where the collection format files are stored. Defaults to FORMATS_DIR.

    Returns
    -------
    dict
        The collection format in JSON format.

    Raises
    ------
    FileNotFoundError
        If the collection format file does not exist.

    """
    # https://github.com/stac-utils/stac-sentinel/blob/main/stac_sentinel/sentinel-s2-l2a.json
    # https://github.com/appelmar/gdalcubes/blob/master/inst/formats/Sentinel2_L2A_THEIA.json
    file_path = Path(formats_dir).expand().relpath() / type+".json"
    with open(file_path) as f:
        fmt = json.load(f)
        return fmt

def stac_asset_info_from_raster(band_file, band_fmt=None):
    """Parse band information for stac.
    
    It uses the file basename to get the band and 
    the rasterio header info for the rest.

    Parameters
    ----------
    band_file : str
        Path to the band file.
    band_fmt : dict, optional
        The band format information, by default None.
        See `collection_format`.
    
    Returns
    -------
    dict
        The stac asset information.
    """
    
    # band_fmt = fmt["bands"]["B02"]

    band_file = Path(band_file).expand()
    
    # get rasterio information
    if band_fmt is None:
        band_fmt = dict(roles=["data"])

    raster_data = any([r in ["reflectance", "data", "overview"] for r in band_fmt["roles"]])
    if raster_data:
        bbox, media_type, gsd, meta, tags, scales, offsets = get_rio_info(band_file)
    else:
        media_type = "application/"+band_file.ext[1:]

    # build stac information
    stac_fields = {
        "href": band_file,
        "type" : media_type,
    }

    band_fmt = {k:v for k,v in band_fmt.items() if k not in ["pattern", "nodata"]}

    stac_fields.update(band_fmt)

    # add projection as the resolution is not the same for all bands
    # It could be set at the item level otherwise.
    proj_info = stac_proj_info(bbox, gsd, meta)
    stac_fields.update(proj_info)
    raster_info = stac_raster_info(meta, tags, gsd, scales, offsets)
    stac_fields.update(raster_info)
    
    return stac_fields

def properties_from_assets(assets, update_assets=True):
    """
    Get the bbox (WGS84), the geometry (WGS84) and
    the unique proj::epsg property from assets.

    Parameters
    ----------
    assets : dict of pystac.Asset
        Dict of assets.
    update_assets : bool, optional
        Removes the proj::epsg from assets if unique. Defaults to True.

    Returns
    -------
    tuple
        Bounding box in WGS84, WGS84 geometry in GeoJSON, and properties.
    """
    properties = {}
    assets = [(k, v) for k, v in assets.items()]
    df_assets = DataFrame(assets, columns=["key", "asset"])
    epsg_list = df_assets["asset"].apply(lambda x: x.extra_fields["proj:epsg"])
    bbox_list = df_assets["asset"].apply(lambda x: box(*x.extra_fields["proj:bbox"]))
    if len(epsg_list.unique()) == 1:
        properties.update({
            "proj:epsg" : int(epsg_list[0])
        })

        if update_assets:
        # remove epsg from extra_fields
            df_assets["asset"].apply(lambda x: x.extra_fields.pop("proj:epsg"))

    g = unary_union([gpd.GeoSeries(bbox, crs=epsg).to_crs(4326).geometry for bbox, epsg in zip(bbox_list, epsg_list)])
    bbox_wgs = list(g.bounds)
    geometry = json.loads(to_geojson(g))
    return bbox_wgs, geometry, properties



def stac_item_parser(item_dir, fmt, assets=None, expand_end_date=True):
    """Parse the item information from the scene directory.

    Parameters
    ----------
    item_dir : str
        The directory path of the scene.
    fmt : dict
        The format of the images.
        See `collection_format`.
    assets : dict, optional
        The assets information, by default None.
        See `stac_asset_info_from_raster`.
    expand_end_date : bool, optional
        Whether to expand the end_date to the last second of the day, by default True.
        At the moment, the STAC specs considers end_datetime as inclusive, which means that
        if end date is 2019-12-31, it should default to 2019-12-31T23:59:59.999999999Z.
        We simplify it to 2019-12-31T23:59:59Z.
        See https://github.com/radiantearth/stac-spec/issues/1255.
    
    Returns
    -------
    dict
        The STAC item information.
    
    Examples
    --------
    >>> from path import Path
    >>> from simplestac.local import collection_format, stac_item_parser
    >>> item_dir = Path.cwd() / "data" / "s2_scenes" / "SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1"
    >>> fmt = collection_format("S2_L2A_THEIA")
    >>> item = stac_item_parser(item_dir, fmt)
    """

    item_dir = Path(item_dir).expand()
    fmt = fmt["item"]

    # parsing properties
    dt_dict = {} # datetime
    properties={}
    if "properties" in fmt:
        properties = dict()
        for k, v in fmt["properties"].items():
            if isinstance(v, str):
                properties[k] = v
            if "pattern" in v:
                match = re.match(v["pattern"], item_dir.name)
                if match is not None:
                    s = match.group(1)
                    if k.endswith("datetime") and "format" in v:
                        dt_dict[k] = to_datetime(s, format=v["format"])
                    else:
                        properties[k] = s
                        # dt_dict[k] = dt # str(dt)+"Z"

    # datetime defined at item level for retro-compatibility
    for k,v in fmt.items():
        if k.endswith("datetime"):
            if "pattern" in v:
                dt = re.match(v["pattern"], item_dir.name)
                if dt is not None:
                    dt = to_datetime(dt.group(1), format=v["format"])
                    dt_dict[k] = dt
    
    # have end_datetime inclusive
    if expand_end_date:
        if "end_datetime" in dt_dict:
            dt_dict["end_datetime"] = to_datetime(dt_dict["end_datetime"].date()) + timedelta(days=1, seconds= -1)

    # parsing id, default is the image directory name
    if "id" in fmt and "pattern" in fmt["id"]:
        match = re.match(fmt["id"]["pattern"], item_dir.name)
        if match is None:
            raise(Exception(f"Pattern {fmt['id']['pattern']} not found in {item_dir}"))
        id = match.group(1)
    else:
        id = item_dir.name
    stac_fields = dict(
        id = id,
        datetime=None, # necessary for pystac.Item if datetime is in properties
        properties = properties,
        stac_extensions = [eo.SCHEMA_URI, projection.SCHEMA_URI, raster.SCHEMA_URI],
    )
    stac_fields.update(dt_dict)

    ### common to any other item ###
    geometry = bbox_wgs = None
    if assets is not None:
        bbox_wgs, geometry, assets_props = properties_from_assets(assets)
        
        # In case of ItemCollection, href is not included as it designates the json file
        # where the item description should be saved. Using pystac Collection or Catalog,
        # the href would be filled when saved.
        stac_fields.update(dict(
            assets = assets,
            bbox = bbox_wgs, # converts tuple to list
            geometry = geometry,
        ))
        stac_fields["properties"].update(assets_props)

        return stac_fields

def stac_asset_parser(item_dir, fmt):
    """Parse the asset information from the scene directory.

    Parameters
    ----------
    item_dir : str
        The directory path of the scene.
    fmt : dict
        Containing item "item_assets" with the asset format of the scene.
        See `collection_format`.

    Returns
    -------
    dict
        The STAC asset information.
    
    Examples
    --------
    >>> from path import Path
    >>> from simplestac.local import collection_format, stac_asset_parser
    >>> item_dir = Path.cwd() / "data" / "s2_scenes" / "SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1"
    >>> fmt = collection_format("S2_L2A_THEIA")
    >>> assets = stac_asset_parser(item_dir, fmt)
    """
    item_dir = Path(item_dir).expand()
    fmt = fmt["item_assets"]

    # parsing assets
    asset_list = []
    all_files = [f for f in item_dir.walkfiles()]
    bands = fmt
    for key, band in bands.items():
        if "pattern" not in band:
            continue
        band_files = [f for f in all_files if re.match(band["pattern"]+"$", f.name)]
        if len(band_files)==0:
            logger.debug(f"Band '{key}' not found in {item_dir}")
            continue
        stac_info = stac_asset_info_from_raster(band_files[0], band)
        asset_list.append((key, pystac.Asset.from_dict(stac_info)))
    
    # sort assets in the order of theia_band_index
    df =  DataFrame(asset_list, columns=['band', 'asset'])
    assets = list(df.itertuples(index=False, name=None))
    assets = {k:v for k,v in assets}

    return assets



class MyStacItem(object):
    """Create a STAC item from a local directory.
    
    Parameters
    ----------
    fmt : dict
        The format of the item to parse.
        See `collection_format`.
    item_parser : function
        The function to parse the item information.
        See `stac_item_parser` for an example.
    asset_parser : function
        The function to parse the asset information.
        See `stac_asset_parser` for an example.

    Examples
    --------
    >>> from path import Path
    >>> from simplestac.local import collection_format, MyStacItem
    >>> item_dir = Path.cwd() / "data" / "s2_scenes" / "SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1"
    >>> fmt = collection_format("S2_L2A_THEIA")
    >>> item_creator = MyStacItem(fmt)
    >>> item = item_creator.create_item(item_dir)
    """

    def __init__(self, fmt, item_parser=stac_item_parser, asset_parser=stac_asset_parser):
        """
        Initializes a new instance of item creator.

        Parameters
        ----------
        fmt : dict
            The format of the item to parse.
            See `collection_format`.
        item_parser : function
            The function to parse the item information.
            See `stac_item_parser` for an example.
        asset_parser : function
            The function to parse the asset information.
            See `stac_asset_parser` for an example.
        """
        # if item_dir is not None:
        #     self.item_dir = item_dir
        self.fmt = fmt
        self.item_parser = item_parser
        self.asset_parser = asset_parser
    
    @property
    def item_dir(self):
        return self._item_dir
    
    @item_dir.setter
    def item_dir(self, x):
        self._item_dir = Path(x).expand()
    
    def create_item(self, item_dir, validate=True):
        """Create the item for the scene.
        
        Parameters
        ----------
        item_dir : str
            The directory path of the scene.
        validate : bool, optional
            Whether to validate the item structure, by default True
        
        Returns
        -------
        pystac.Item
            The created item
        """

        self.item_dir = item_dir
        # create assets
        assets = self.asset_parser(self.item_dir, self.fmt)
        # prepare item dict
        # stac_info = self.get_item_info(assets)
        stac_info = self.item_parser(self.item_dir, self.fmt, assets)
        
        # create item
        item = pystac.Item(**stac_info)
        # validate item structure        
        if validate:
            item.validate()

        return item


def get_item_dirs(input_dir, fmt):
    """
    Recursively retrieves item directories based on the input directory and format.

    Parameters
    ----------
    input_dir : str
        The input directory to search for item directories.
    fmt : dict
        The format containing the pattern for item directories.
        See `collection_format`.
    Returns
    -------
    list
        A list of item directories found based on the format.
    """    
    item_dirs = []
    if isinstance(input_dir, list):
        for d in input_dir:
            item_dirs.extend(get_item_dirs(d, fmt))
        return item_dirs

    input_dir = Path(input_dir).expand()
    if re.match(fmt["item"]["pattern"], input_dir.name):
        item_dirs.append(input_dir)
    else:
        item_dirs = get_item_dirs(input_dir.dirs(), fmt)
    return item_dirs
    
def build_item_collection(input_dir,
                          fmt, 
                          item_parser=stac_item_parser,
                          asset_parser=stac_asset_parser,
                          progress=True, validate=True, **kwargs):
    """Build an item collection with the scenes in an input_dir,
    using fmt for parsing items and assets information.

    Parameters
    ----------
    input_dir : str | List[str]
        Path to the input directory containing the Theia image files.
        A list of directories is also possible.
    fmt : dict
        See `collection_format`.
    item_parser : function, optional
        The function to parse the item information, by default stac_item_parser.
    progress : bool, optional
        Whether to show a progress bar, by default True
    validate : bool, optional
        Whether to validate the item structure, by default True.
        Warning: this adds a lot of overhead (x10) so disable it if not needed.
    **kwargs : dict, optional
        Additional keyword arguments passed to pystac.ItemCollection

    Returns
    -------
    pystac.ItemCollection
        This collection can then be saved into a unique json file
    
    Examples
    --------
    >>> from simplestac import build_item_collection
    >>> from pathlib import Path
    >>> input_dir = Path.cwd() / "data" / "s2_scenes"
    >>> fmt = collection_format("S2_L2A_THEIA")
    >>> col = build_item_collection(input_dir, fmt)
    """
    from simplestac.utils import ItemCollection # avoids circular import
    
    item_dirs = get_item_dirs(input_dir, fmt)
    
    if len(item_dirs) == 0:
        logger.warning(f"No item found in {input_dir}")
        return
    
    items = []
    logger.info("Building item collection...")
    item_creator = MyStacItem(fmt, item_parser=item_parser, asset_parser=asset_parser)
    for item in tqdm(item_dirs, disable=not progress):
        items.append(
            item_creator.create_item(item, validate=validate)
        )
    return ItemCollection(items, clone_items=False, **kwargs)  