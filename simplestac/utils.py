""""Notes:
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
## TODO: see if rio-cogeo would simplify the code


import json
import pystac
import geopandas as gpd
from pystac.item_collection import ItemLike
import rasterio
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
from rasterio.vrt import WarpedVRT
from path import Path
import re
from pandas import DataFrame, to_datetime
import pandas as pd
from pystac.media_type import MediaType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pystac.extensions import eo, projection
import stac_static

from shapely.geometry import box
from shapely import to_geojson
from shapely.ops import unary_union

import warnings
import stackstac
from stac_static.search import to_geodataframe
import xarray as xr
import numpy as np
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm

#### Generic functions and classes ####



class ExtendPystacClasses:
    """Add capacities to_xarray and filter to pystac Catalog, Collection, ItemCollection"""

    def to_xarray(self, xy_coords="center", bbox=None, geometry=None, **kwargs):
        """Returns a DASK xarray()
        
        This is a proxy to stackstac.stac
        
        Arguments are:
        assets=frozenset({'image/jp2', 'image/tiff', 'image/vnd.stac.geotiff', 'image/x.geotiff'}),
        epsg=None, resolution=None, bounds=None, bounds_latlon=None,
        snap_bounds=True, resampling=Resampling.nearest, chunksize=1024,
        dtype=dtype('float64'), fill_value=nan, rescale=True,
        sortby_date='asc', xy_coords='center', properties=True,
        band_coords=True, gdal_env=None,
        errors_as_nodata=(RasterioIOError('HTTP response code: 404'), ),
        reader=<class 'stackstac.rio_reader.AutoParallelRioReader'>

        For details, see [stackstac.stac](https://stackstac.readthedocs.io/en/latest/api/main/stackstac.stack.html)

        Notes:
        ------
        Here, xy_coords="center" is the default to be consistent with rioxarray,
        cf https://github.com/gjoseph92/stackstac/issues/207. Otherwise, stackstac.stac has
        xy_coords="topleft" as the default.


        """
        # We could also have used :
        # stackstac.stack(self, xy_coords=xy_coords, bounds=list(bbox), **kwargs)
        # site-packages/stackstac/prepare.py:364: UserWarning: The argument 'infer_datetime_format' is deprecated and will be removed in a future version. A strict version of it is now the default, see https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. You can safely remove this argument.
        # times = pd.to_datetime(
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            arr = stackstac.stack(self, xy_coords=xy_coords, **kwargs)

        if bbox is not None:
            arr = arr.rio.clip_box(*bbox)
        if geometry is not None:
            arr = arr.rio.clip(geometry)
        return arr
    
    def filter(self, asset_names=None, clone_items=True, **kwargs):
        """Filter items with stac-static search.
        
        Documentation copied from stac-static.

        All parameters correspond to query parameters described in the `STAC API - Item Search: Query Parameters Table
        <https://github.com/radiantearth/stac-api-spec/tree/master/item-search#query-parameter-table>`__
        docs. Please refer to those docs for details on how these parameters filter search results.

        Args:
            ids: List of one or more Item ids to filter on.
            collections: List of one or more Collection IDs or :class:`pystac.Collection`
                instances. Only Items in one
                of the provided Collections will be searched
            bbox: A list, tuple, or iterator representing a bounding box of 2D
                or 3D coordinates. Results will be filtered
                to only those intersecting the bounding box.
            intersects: A string or dictionary representing a GeoJSON geometry, or
                an object that implements a
                ``__geo_interface__`` property, as supported by several libraries
                including Shapely, ArcPy, PySAL, and
                geojson. Results filtered to only those intersecting the geometry.
            datetime: Either a single datetime or datetime range used to filter results.
                You may express a single datetime using a :class:`datetime.datetime`
                instance, a `RFC 3339-compliant <https://tools.ietf.org/html/rfc3339>`__
                timestamp, or a simple date string (see below). Instances of
                :class:`datetime.datetime` may be either
                timezone aware or unaware. Timezone aware instances will be converted to
                a UTC timestamp before being passed
                to the endpoint. Timezone unaware instances are assumed to represent UTC
                timestamps. You may represent a
                datetime range using a ``"/"`` separated string as described in the spec,
                or a list, tuple, or iterator
                of 2 timestamps or datetime instances. For open-ended ranges, use either
                ``".."`` (``'2020-01-01:00:00:00Z/..'``,
                ``['2020-01-01:00:00:00Z', '..']``) or a value of ``None``
                (``['2020-01-01:00:00:00Z', None]``).

                If using a simple date string, the datetime can be specified in
                ``YYYY-mm-dd`` format, optionally truncating
                to ``YYYY-mm`` or just ``YYYY``. Simple date strings will be expanded to
                include the entire time period, for example:

                - ``2017`` expands to ``2017-01-01T00:00:00Z/2017-12-31T23:59:59Z``
                - ``2017-06`` expands to ``2017-06-01T00:00:00Z/2017-06-30T23:59:59Z``
                - ``2017-06-10`` expands to ``2017-06-10T00:00:00Z/2017-06-10T23:59:59Z``

                If used in a range, the end of the range expands to the end of that
                day/month/year, for example:

                - ``2017/2018`` expands to
                ``2017-01-01T00:00:00Z/2018-12-31T23:59:59Z``
                - ``2017-06/2017-07`` expands to
                ``2017-06-01T00:00:00Z/2017-07-31T23:59:59Z``
                - ``2017-06-10/2017-06-11`` expands to
                ``2017-06-10T00:00:00Z/2017-06-11T23:59:59Z``

            filter: JSON of query parameters as per the STAC API `filter` extension
            filter_lang: Language variant used in the filter body. If `filter` is a
                dictionary or not provided, defaults
                to 'cql2-json'. If `filter` is a string, defaults to `cql2-text`.
            
        Notes:
            Argument filter would search into the first level of metadata of the asset.
            If the metadata to filter is a string, it should be used as 
            a string into a string, examples:
             - filter="constellation = 'sentinel-2' and tilename = 'T31UFQ'"
             - filter="tilename in ('T31UFQ', 'T31UFQ')"
            
             In order to filter/select assets, use to_xarray(asset=...) or to_xarray().sel(band=...)
        """
        if not clone_items:
            x = self.to_geodataframe(include_items=True)
            res = stac_static.search(x, **kwargs)
            # TODO check what happens if no items correspond to the query
            res = self.__class__(res.result.item.to_list(), clone_items=False)
        else:    
            res = self.__class__(stac_static.search(self, **kwargs).item_collection())

        if asset_names is not None:
            for item in res.items:
                item.assets = {k:a for k, a in item.assets.items() if k in asset_names}
        
        return res

    def to_geodataframe(self, include_items=False, **kwargs):
        """
        Convert the current pystac object to a GeoDataFrame.

        Parameters
        ----------
        include_items : bool, optional
            Whether to include the items in the resulting GeoDataFrame.
            Defaults to False.

        Returns
        -------
        GeoDataFrame
            The converted GeoDataFrame.

        Other Parameters
        ----------------
        **kwargs
            Additional keyword arguments passed to stac_static.to_geodataframe.

        See Also
        --------
        stac_static.to_geodataframe
        """
        res = to_geodataframe(self)
        if include_items:
            items = pd.DataFrame([(x.id, x) for x in self.items], columns=["id", "item"])
            res = res.merge(items, on="id")
        return res

    def sort_items(self, inplace=False, **kwargs):
        """Sorts the items in the collection

        Parameters
        ----------
        inplace : bool
            Whether to sort the collection by reference.
        **kwargs
            Additional keyword arguments passed to pandas.DataFrame.sort_values

        Returns
        -------
        object
            A clone of the collection if inplace is False, otherwise None.
            
        """
        if inplace:
            x = self
        else:
            x = self.clone()
        df = x.to_geodataframe(include_items=True)
        df.sort_values(inplace=True, axis=0, **kwargs)
        x.items = df.item.to_list()
        if not inplace:
            return x
    
    def apply_items(self, fun,
                    name,
                    output_dir,
                    overwrite=False,
                    inplace=False,
                    datetime=None,
                    bbox=None,
                    geometry=None,
                    progress=True,
                    **kwargs):
        """
        Apply a given function to each item in the collection,
        save the result in a raster file, 
        and add the new asset to the corresponding item.

        Parameters
        ----------
        fun : callable
            The function to apply to each item.
        name : str
            The name of the new asset. 
            This also serves as the file name suffix: "{item.id}_{name}.tif"
        output_dir : str
            The directory where the output will be saved. Created if it does not exist.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to False.
        inplace : bool, optional
            Whether to modify the collection in place. Defaults to False.
            In that case, a cloned collection is returned.
        bbox : tuple, optional
            A bounding box to clip_box the items with. Defaults to None.
        geometry : shapely.geometry, optional
            A geometry to clip the items with. Defaults to None.
        progress : bool, optional
            Whether to show a progress bar. Defaults to True.
        **kwargs
            Additional keyword arguments to pass to the function.

        Returns
        -------
        object
            A clone of the collection if inplace is False, otherwise None.
        """        
        # could be a method added to item or collection
        if inplace:
            x = self
        else:
            x = self.clone()
        
        if datetime is not None:
            x = x.filter(datetime=datetime, clone_items=False)

        for item in tqdm(x.items, disable=not progress):
            apply_item(item, fun, name=name, output_dir=output_dir,
                            overwrite=overwrite, copy=False, 
                            bbox=bbox, geometry=geometry,
                            **kwargs)
        if not inplace:
            return x

    def apply_rolling(self, fun, 
                      name, 
                      output_dir,
                      overwrite=False,
                      window=2,
                      inplace=False,
                      datetime=None,
                      bbox=None,
                      geometry=None,
                      progress=True,
                      **kwargs):
        """
        Apply a rolling window function to the items in the object,
        save the result in a raster file, 
        and add it as a new asset to the corresponding item.

        Parameters
        ----------
        fun : callable
            The function to apply to each item.
        name : str
            The name of the new asset. 
            This also serves as the file name suffix: "{item.id}_{name}.tif"
        output_dir : str
            The directory where the output will be saved. Created if it does not exist.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to False.
        inplace : bool, optional
            Whether to modify the collection in place. Defaults to False.
            In that case, a cloned collection is returned.
        bbox : tuple, optional
            A bounding box to clip_box the items with. Defaults to None.
        geometry : shapely.geometry, optional
            A geometry to clip the items with. Defaults to None.
        progress : bool, optional
            Whether to show a progress bar. Defaults to True.
        **kwargs
            Additional keyword arguments to pass to the function.

        Returns
        -------
        object
            A clone of the collection if inplace is False, otherwise None.
        """        

        if inplace:
            x = self
        else:
            x = self.clone()

        if datetime is not None:
            x = x.filter(datetime=datetime, clone_items=False)

        # case of multiple outputs
        if isinstance(name, str):
            name = [name]
        if isinstance(output_dir, str):
            output_dir = [output_dir]
        if len(name) != len(output_dir):
            raise ValueError("output_dir must have the same length as name")
        Nout = len(name)
        
        output_dir = [Path(d).expand().mkdir_p() for d in output_dir] # make sure they exist 
        for i in tqdm(np.arange(window-1, len(x.items)), disable=not progress):
            subitems = x.items[max((i-window),0):i]
            subcol = self.__class__(subitems, clone_items=False)
            raster_file = [d / f"{subitems[-1].id}_{n}.tif" for n, d in zip(name, output_dir)]
            if not overwrite and all([r.exists() for r in raster_file]):
                logger.debug(f"File already exists, skipping computation: {raster_file}")
                res = tuple([None]*Nout)
            else:
                # compute fun
                res = fun(subcol.to_xarray(bbox=bbox, geometry=geometry), **kwargs)
                if not isinstance(res, tuple):
                    res = (res,)
                if len(res) != Nout:
                    raise ValueError(f"Expected {Nout} outputs, got {len(res)}")
                for r,f in zip(res, raster_file):
                    if r is None:
                        continue
                    # write result
                    logger.debug("Writing: ", f)
                    write_raster(r, f, overwrite=overwrite)
            for n, f in zip(name, raster_file):
                if f.exists():
                    stac_info = stac_asset_info_from_format(f)
                    asset = pystac.Asset.from_dict(stac_info)
                    subitems[-1].add_asset(key=n, asset=asset)
        
        if not inplace:
            return x


class ItemCollection(pystac.ItemCollection, ExtendPystacClasses):
    pass

class Catalog(pystac.Catalog, ExtendPystacClasses):
    pass

class Collection(pystac.Collection, ExtendPystacClasses):
    pass


def write_raster(x: xr.DataArray, file, driver="COG", overwrite=False):
    if Path(file).exists() and not overwrite:
        logger.debug(f"File already exists, skipped: {file}")
        return
    if x.dtype == 'bool':
        x = x.astype('uint8')
    x.rio.to_raster(file, driver=driver)


def apply_item(x, fun, name, output_dir, overwrite=False,
               copy=True, bbox=None, geometry=None, **kwargs):

# could be a method added to item or collection
    if not isinstance(x, pystac.Item):
        raise ValueError("x must be a pystac.Item")
    
    if copy:
        x = x.clone()


    if isinstance(name, str):
        name = [name]
    if isinstance(output_dir, str):
        output_dir = [output_dir]
    if len(name) != len(output_dir):
        raise ValueError("output_dir must have the same length as name")
    Nout = len(name)
    output_dir = [Path(d).expand().mkdir_p() for d in output_dir] 
    
    raster_file = [d / f"{x.id}_{n}.tif" for n, d in zip(name, output_dir)]
    if not overwrite and all([r.exists() for r in raster_file]):
        logger.debug(f"File already exists, skipping computation: {raster_file}")
        res = tuple([None]*Nout)
    else:
        # compute fun
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            arr = stackstac.stack(x, xy_coords="center")
        if bbox is not None:
            arr = arr.rio.clip_box(*bbox)
        if geometry is not None:
            arr = arr.rio.clip(geometry)
        res = fun(arr, **kwargs)

        if not isinstance(res, tuple):
            res = (res,)
        if len(res) != Nout:
            raise ValueError(f"Expected {Nout} outputs, got {len(res)}")
        for r,f in zip(res, raster_file):
            if r is None:
                continue
            # write result
            logger.debug("Writing: ", f)
            write_raster(r, f, overwrite=overwrite)
    for n, f in zip(name, raster_file):
        if f.exists():
            stac_info = stac_asset_info_from_format(f)
            asset = pystac.Asset.from_dict(stac_info)    
            x.add_asset(key=n, asset=asset)
    return x

#######################################

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

############# Build collection from json
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

def stac_asset_info_from_format(band_file, band_fmt=None):
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
            stac_info = stac_asset_info_from_format(band_files[0], band)
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