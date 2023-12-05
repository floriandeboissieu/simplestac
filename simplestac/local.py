""""
This module aims at providing a simple interface to build a STAC ItemCollection
from a series of scenes/images in local files.


Notes:
------
    A STAC collection or a catalog is a nested dict (json) listing a number of items
    (e.g. a remote sensing acquisition) and their assets (e.g. bands)

    A tutorial on Landsat, but transposable for Sentinel or others is available here:
    https://pystac.readthedocs.io/en/stable/tutorials/creating-a-landsat-stac.html
    
    # Items

    The required fields for an item (without type and stac_version that are automatically filled by pystac):
        - id: name underwhich the item will be registered
        - geometry: a geojson geometry in dict, see shapely.to_geojson followed by json.loads
        - bbox (List[float]): the bounding box of the item
        - properties (dict): with at least field "datetime" for the date of acquisition.
        - stac_extensions (List[str]): list of the schema URIs used for validation,
        e.g. pystac.extensions.eo.SCHEMA_URI

    # Assets

    The required fields for a stac asset is the `href` (i.e. the path to the file, e.g. a raster file) and
    the `key` (usually the band name) under which it will be registered in the Item.

    Optional but recommended:
        - title (str): usually the band name
        - type (str): see `list(pystac.MediaType)` or
        https://github.com/radiantearth/stac-spec/blob/master/best-practices.md#working-with-media-types
        - roles (List[str]): thumbnail, overview, data, metadata
        - extra_fields (dict): all the rest of the metadata, in particular exetnsion fields

    Other fields are also recommended, especially for satellite data:
    https://github.com/radiantearth/stac-spec/blob/master/item-spec/common-metadata.md
    and https://github.com/radiantearth/stac-spec/blob/master/item-spec/common-metadata.md#instrument

    With the Electro-Optical extension the additionnal fields (extra_fields) are
        - eo:bands (List[dict]): each item of the list corresponds to a band of the band file
        see https://github.com/stac-extensions/eo/#band-object for specs (names and types)
            - name (str): band name
            - common_name (str): the color name for the reflectance bands
            - description (str)
            - center_wavelength (number)
            - full_width_half_max (number)
            - solar_illumination (number)
    
    With the Projection extension the additionnal fields are:
    See https://github.com/stac-extensions/projection/#projection-extension-specification
    and https://github.com/stac-extensions/projection/#best-practices
        - proj:epsg
        - proj:wkt2
        - proj:projjson
        - proj:geometry
        - proj:bbox
        - proj:centroid (dict(lat, lon))
        - proj:shape (List[height, width]): https://github.com/stac-extensions/projection/#projshape
        - proj:transform
    The projection information can be set at the item level if it is the same for
    all bands. Recommendations are to set it at the asset level if it differs from
    
    Although the extension fields are all "optional", if used it must be in the good format,
    the item validation fails otherwise.

    Other extensions are available although not used here,
    see https://github.com/radiantearth/stac-spec/tree/master/extensions#stable-stac-extensions and
    https://stac-extensions.github.io/
"""
## TODO: see if rio-cogeo would simplify the code and externalise
# the parsing of the metadata from the band files.


import json
import geopandas as gpd
import logging
from path import Path
import rasterio
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
from rasterio.vrt import WarpedVRT
import re
import pandas as pd
from pandas import DataFrame, to_datetime
import pystac
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pystac.extensions import eo, projection
from shapely.geometry import box
from shapely import to_geojson
from shapely.ops import unary_union
from tqdm import tqdm
import warnings


logger = logging.getLogger(__name__)

############# Generic functions to build collection
def get_rio_info(band_file):
    """
    Get information about a raster band file.

    Parameters
    ----------
    band_file : str
        The path to the raster band file.

    Returns
    -------
    tuple
        A tuple containing the bounding box of the band file, the media type,
        the ground sample distance (gsd), and the metadata.
    """
    # band_file = "~/git/fordeadv2/fordead_data-main/sentinel_data/dieback_detection_tutorial/study_area/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1/SENTINEL2A_20151203-105818-575_L2A_T31UFQ_D_V1-1_FRE_B11.tif"
    band_file = Path(band_file).expand()
    with rasterio.open(band_file) as src:
        bbox = src.bounds
        meta = src.meta
        media_type = get_media_type(src)
        gsd = src.res[0]
    # If needed at some point, we could test MediaType.COG with rio_cogeo.cogeo.cog_validate
    # from rio_cogeo.cogeo import cog_validate
    # iscog, _, _ = cog_validate(band_file)
    # if iscog:
    #     media_type = pystac.MediaType.COG
        
    return bbox, media_type, gsd, meta

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
    _, _, centroid = bbox_to_wgs(bbox, meta["crs"])
    proj = dict(
        epsg = epsg,
        wkt2 = meta["crs"].to_wkt(),
        # geometry = bbox_to_geom(bbox),
        geometry = json.loads(to_geojson(box(*bbox))),
        centroid = centroid,
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

def bbox_to_wgs(bbox, epsg):
    g = gpd.GeoSeries(box(*bbox), crs=epsg)
    g_wgs = g.to_crs(4326)
    bbox = [float(f) for f in g_wgs.total_bounds]
    geom = json.loads(to_geojson(g_wgs.geometry[0]))
    centroid = g.geometry.centroid.to_crs(4326).iat[0]
    centroid = {"lat": float(centroid.y), "lon": float(centroid.x)}
    return bbox, geom, centroid

#######################################

############# Build collection from json format
FORMATS_DIR = Path(__file__).parent / "formats"

def common_name_table():
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
        bbox, media_type, gsd, meta = get_rio_info(band_file)
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
    
    return stac_fields


class MyStacItem(object):
    """Create a STAC item from a local directory."""

    def __init__(self, image_dir, fmt):
        """
        Initializes a new instance of the class.

        Parameters
        ----------
        image_dir : str
            The directory path of the scene.
        fmt : str
            The format of the images.
            See `collection_format`.

        """
        image_dir = Path(image_dir).expand()
        self.image_dir = image_dir
        self.fmt = fmt
    def get_item_info(self, assets=None):
        """Get the item information as a dictionary in STAC format.

        The information is based on the item directory name.
        It typically search for a datetime pattern and properties defined in `fmt`.

        Parameters
        ----------
        assets : list, optional
            The list of assets, by default None.

        Returns
        -------
        dict
            The STAC item information.
        """
        image_dir = self.image_dir
        image_fmt = self.fmt["item"]
        dt = re.match(image_fmt["datetime"]["pattern"], image_dir.name)
        if dt is not None:
            dt = to_datetime(dt.group(1), format=image_fmt["datetime"]["format"])

        properties={}
        if "properties" in image_fmt:
            properties = dict()
            for k, v in image_fmt["properties"].items():
                if isinstance(v, str):
                    properties[k] = v
                if "pattern" in v:
                    match = re.match(v["pattern"], image_dir.name)
                    if match is not None:
                        properties[k] = match.group(1)
            
            
        id = image_dir.name

        ### common to any other item ###
        geometry = bbox = None
        if assets is not None:
            df_assets = DataFrame(assets, columns=["key", "asset"])
            epsg_list = df_assets["asset"].apply(lambda x: x.extra_fields["proj:epsg"])
            bbox_list = df_assets["asset"].apply(lambda x: box(*x.extra_fields["proj:bbox"]))
            if len(epsg_list.unique()) == 1:
                epsg = epsg_list[0]
                bbox = list(unary_union(bbox_list).bounds)
                bbox_wgs, geometry, _ = bbox_to_wgs(bbox, epsg)
                properties.update({
                    "proj:epsg" : int(epsg)
                })
                # remove epsg from extra_fields
                df_assets["asset"].apply(lambda x: x.extra_fields.pop("proj:epsg"))
            else:
                g = unary_union([gpd.GeoSeries(bbox, crs=epsg).to_crs(4326).geometry for bbox, epsg in zip(bbox_list, epsg_list)])
                bbox = g.bounds
                bbox_wgs, geometry, _ = bbox_to_wgs(bbox, 4326)

        
        # In case of ItemCollection, href is not included as it designates the json file
        # where the item description should be saved. Using pystac Collection or Catalog,
        # the href would be filled when saved.
        stac_fields = dict(
            id = id,
            datetime = dt,
            properties = properties,
            assets = {k:v for k,v in assets},
            bbox = bbox_wgs,
            geometry = geometry,
            stac_extensions = [eo.SCHEMA_URI, projection.SCHEMA_URI]
        )

        return stac_fields


    def create_assets(self):
        """Looks for the bands in the directory and 
        create the asset information for the item."""

        asset_list = []
        all_files = [f for f in self.image_dir.walkfiles()]
        bands = self.fmt["item_assets"]
        for key, band in bands.items():
            if "pattern" not in band:
                continue
            band_files = [f for f in all_files if re.match(band["pattern"]+"$", f.name)]
            if len(band_files)==0:
                logger.debug(f"Band '{key}' not found in {self.image_dir}")
                continue
            stac_info = stac_asset_info_from_raster(band_files[0], band)
            asset_list.append((key, pystac.Asset.from_dict(stac_info)))
        
        # sort assets in the order of theia_band_index
        df =  DataFrame(asset_list, columns=['band', 'asset'])
        assets = list(df.itertuples(index=False, name=None))

        return assets
    
    def create_item(self, validate=True):
        """Create the item for the scene.
        
        Parameters
        ----------
        validate : bool, optional
            Whether to validate the item structure, by default True
        
        Returns
        -------
        pystac.Item
            The created item
        """

        # create assets
        assets = self.create_assets()
        # prepare item dict
        stac_info = self.get_item_info(assets)
        
        # create item
        item = pystac.Item(**stac_info)
        # validate item structure        
        if validate:
            item.validate()

        return item

def build_item_collection(input_dir, fmt, progress=True, **kwargs):
    """Build an item collection with the scenes in an input_dir,
    using fmt for parsing items and assets information.

    Parameters
    ----------
    input_dir : str | List[str]
        Path to the input directory containing the Theia image files.
        A list of directories is also possible.
    fmt : dict
        See `collection_format`.
    progress : bool, optional
        Whether to show a progress bar, by default True
    **kwargs : dict, optional
        Additional keyword arguments passed to pystac.ItemCollection

    Returns
    -------
    pystac.ItemCollection
        This collection can then be saved into a unique json file
    """
    from simplestac.utils import ItemCollection # avoids circular import

    if isinstance(input_dir, list):
        items = []
        for d in input_dir:
            col = build_item_collection(d, fmt, **kwargs)
            items.extend(col.items)
        return ItemCollection(items, clone_items=False, **kwargs)
    
    input_dir = Path(input_dir).expand()
    item_dirs =  [d for d in input_dir.dirs() if re.match(fmt["item"]["pattern"], d.name)]
    items = []
    logger.info("Building item collection...")
    for item in tqdm(item_dirs, disable=not progress):
        items.append(
            MyStacItem(item, fmt).create_item()
        )
    return ItemCollection(items, clone_items=False, **kwargs)  