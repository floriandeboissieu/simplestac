"""This module aims at providing utilities to work with STAC ItemCollection.
"""

import json
import logging
import numpy as np
from path import Path
import pandas as pd
import pystac
from pystac.item_collection import ItemLike
import re
from shapely import from_geojson, to_geojson, intersection
import stac_static
from stac_static.search import to_geodataframe
import stackstac
import xarray as xr
import rioxarray # necessary to activate rio plugin in xarray
from tqdm import tqdm
from typing import Union
import warnings
import datetime

from simplestac.local import stac_asset_info_from_raster, properties_from_assets

logger = logging.getLogger(__name__)


#### Generic functions and classes ####

S2_THEIA_BANDS = [f"B{i+1}" for i in range(12)]+["B8A"]
S2_SEN2COR_BANDS = [f"B{i+1:02}" for i in range(12)]+["B8A"]

class ExtendPystacClasses:
    """Add capacities to_xarray and filter to pystac Catalog, Collection, ItemCollection"""

    def drop_non_raster(self, inplace=False):
        """Drop non raster assets from each item in the collection.
        
        Parameters
        ----------
        inplace : bool
            Whether to modify the collection in place. Defaults to False.
        
        Returns
        -------
        object
            If `inplace` is False, a cloned collection is returned.       
        """
        if inplace:
            x = self
        else:
            x = self.clone()
        
        for item in x.items: 
            drop_assets_without_proj(item, inplace=True)

        if not inplace:
            return x

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

        Also, by default it is sorted by ascending datetime, see sortby_date.


        """
        # We could also have used :
        # stackstac.stack(self, xy_coords=xy_coords, bounds=list(bbox), **kwargs)
        # site-packages/stackstac/prepare.py:364: UserWarning: The argument 'infer_datetime_format' is deprecated and will be removed in a future version. A strict version of it is now the default, see https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. You can safely remove this argument.
        # times = pd.to_datetime(
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                arr = stackstac.stack(self, xy_coords=xy_coords, **kwargs)
            except ValueError as e:
                if "Cannot automatically compute the resolution" in str(e):
                    raise ValueError(str(e)+"\nOr drop non-raster assets from collection with ItemCollection.drop_non_raster()")
                else:
                    raise e

        if bbox is not None:
            arr = arr.rio.clip_box(*bbox)
        if geometry is not None:
            if hasattr(geometry, 'crs') and geometry.crs != arr.rio.crs:
                logger.debug(f"Reprojecting geometry from {geometry.crs} to {arr.rio.crs}")
                geometry = geometry.to_crs(arr.rio.crs)
            arr = arr.rio.clip(geometry)
        return arr
    
    def filter(self, assets=None, with_assets=None, clone_items=True, **kwargs):
        """Filter items with stac-static search.
        Additional args:

            assets: list
                List of assets to keep in items (other assets are droped).
            with_assets: list
                List of mandatory assets to keep items.
            clone_items: bool
                Whether to clone the items before filtering.
            
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

        if with_assets is not None:
            if isinstance(with_assets, str):
                with_assets = [with_assets]
            res.items = [x for x in res.items if set(with_assets).issubset(x.assets)]

        if assets is not None:
            for item in res.items:
                item.assets = {k:a for k, a in item.assets.items() if k in assets}
            # remove None assets
            res.items = [x for x in res.items if len(x.assets)>0]
        return res

    def to_geodataframe(self, include_items=False, wgs84=True, **kwargs):
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

        if not wgs84: # convert to items epsg
            if not "proj:epsg" in res.columns:
                raise ValueError('Attribute "proj:epsg" is missing.')
            
            epsg = res["proj:epsg"].unique()
            if len(epsg) != 1:
                raise ValueError('Attribute "proj:epsg" is not unique.')
            
            epsg = epsg[0]
            res = res.to_crs(epsg=epsg)
            
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
    
    def drop_duplicates(self, inplace=False):
        """
        A function to drop duplicates from the collection
        based on the item ids.

        Parameters
        ----------
        inplace : bool, optional
            If True, the collection is modified in place.

        Returns
        -------
        ItemCollection
            The collection with duplicates dropped, if inplace is False.
        
        Notes
        -----
        Duplicates seem to be occuring at search depending on the paging.
        See https://github.com/microsoft/PlanetaryComputer/issues/163
        """
        x=self
        if not inplace:
            x=self.clone()

        index = pd.Series([i.id for i in self]).duplicated()
        if index.any():
            x.items = [i for i, v in zip(x.items, ~index) if v]
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
                    writer_args=None,
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
        writer_args : dict or list of dict, optional
            Additional keyword arguments to pass to writer_raster.
        progress : bool, optional
            Whether to show a progress bar. Defaults to True.
        **kwargs
            Additional keyword arguments passed to function `fun`.

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
                            writer_args=writer_args,
                            **kwargs)
        if not inplace:
            return x

    def apply_rolling(self, fun, 
                      name, 
                      output_dir,
                      overwrite=False,
                      window=2,
                      inplace=False,
                      center=False,
                      datetime=None,
                      bbox=None,
                      geometry=None,
                      writer_args=None,
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
        center : bool, optional
            Whether to center the window.
            Defaults to False (the rightmost item will be filled with the new asset).
        bbox : tuple, optional
            A bounding box to clip_box the items with. Defaults to None.
        geometry : shapely.geometry, optional
            A geometry to clip the items with. Defaults to None.
        writer_args : dict or list of dict, optional
            Additional keyword arguments to pass to writer_raster. See `apply_item`.
        progress : bool, optional
            Whether to show a progress bar. Defaults to True.
        **kwargs
            Additional keyword arguments to pass to function `fun`.

        Returns
        -------
        object
            A clone of the collection if inplace is False, otherwise None.
        """        
        # TODO: could be done with padnasDataFrame.rolling function,
        # which would make the things easier
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
        if writer_args is None:
            writer_args = [{}]
        if isinstance(writer_args, dict):
            writer_args = [writer_args]

        if len(name) != len(output_dir):
            if len(output_dir)==1:
                output_dir = output_dir*len(name)
            else:
                raise ValueError("Argument `output_dir` must have length 1 or the same length as `name`.")

        if len(name) != len(writer_args):
            if len(writer_args)==1:
                writer_args = writer_args*len(name)
            else:
                raise ValueError("Argument `writer_args` must have length 1 or the same length as `name`.")
        
        Nout = len(name)
        
        output_dir = [Path(d).expand().mkdir_p() for d in output_dir] # make sure they exist 
        for i in tqdm(range(len(x.items)), disable=not progress):
            subitems = x.items[max((i-window+1),0):i+1]
            if center:
                subitems = x.items[max(i-window//2,0):i+(window-1)//2+1]

            subcol = self.__class__(subitems, clone_items=False)
            raster_file = [d / f"{subitems[-1].id}_{n}.tif" for n, d in zip(name, output_dir)]
            if not overwrite and all([r.exists() for r in raster_file]):
                logger.debug(f"File already exists, skipping computation: {raster_file}")
                res = tuple([None]*Nout)
            else:
                # compute fun
                with xr.set_options(keep_attrs=True):
                    res = fun(subcol.to_xarray(bbox=bbox, geometry=geometry), **kwargs)
                if not isinstance(res, tuple):
                    res = (res,)
                if len(res) != Nout:
                    raise ValueError(f"Expected {Nout} outputs, got {len(res)}")
                for n,r,f,wa in zip(name, res, raster_file, writer_args):
                    if r is None:
                        continue
                    # write result
                    logger.debug("Writing: ", f)
                    r.name = n
                    write_raster(r, f, overwrite=overwrite, **wa)
                    
            for n, f in zip(name, raster_file):
                if f.exists():
                    stac_info = stac_asset_info_from_raster(f)
                    asset = pystac.Asset.from_dict(stac_info)
                    x.items[i].add_asset(key=n, asset=asset)
        
        if not inplace:
            return x


class ItemCollection(pystac.ItemCollection, ExtendPystacClasses):
    pass

# class Catalog(pystac.Catalog, ExtendPystacClasses):
#     pass

# class Collection(pystac.Collection, ExtendPystacClasses):
#     pass

DEFAULT_REMOVE_PROPS = ['.*percentage', 'eo:cloud_cover', '.*mean_solar.*']

def write_assets(x: Union[ItemCollection, pystac.Item],
                 output_dir: str, bbox=None, update=True,
                 xy_coords='center', 
                 remove_item_props=DEFAULT_REMOVE_PROPS,
                 overwrite=False,
                 progress=True,
                 **kwargs):
    """
    Writes item(s) assets to the specified output directory.

    Each item assets is written to a separate raster file with
    path output_dir/item.id/href.name.

    Parameters
    ----------
    x : Union[ItemCollection, pystac.Item]
        The item or collection of items to write assets from.
    output_dir : str
        The directory to write the assets to.
    bbox : Optional
        The bounding box to clip the assets to.
    remove_item_props : list of str
        List of regex patterns to remove from item properties.
        If None, no properties are removed.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    **kwargs
        Additional keyword arguments passed to write_raster.

    """    
    if isinstance(x, pystac.Item):
        x = [x]

    output_dir = Path(output_dir).expand()
    items = []
    for item in tqdm(x, disable=not progress):
        ic = ItemCollection([item], clone_items=True)
        arr = ic.to_xarray(bbox=bbox, xy_coords=xy_coords).squeeze("time")
        item_dir = (output_dir / item.id).mkdir_p()
        for b in arr.band.values:
            filename = '_'.join([item.id, b+'.tif'])
            file = item_dir / f"{filename}"
            try:
                if file.exists() and not overwrite:
                    logger.debug(f"File already exists, skipping asset: {file}")
                else:
                    write_raster(arr.sel(band=b), file, **kwargs)
                
                # update stac asset info            
                stac_info = stac_asset_info_from_raster(file)
                if update:
                    asset_info = item.assets[b].to_dict()
                    asset_info.update(stac_info)
                    stac_info = asset_info
                asset = pystac.Asset.from_dict(stac_info)
                item.add_asset(key=b, asset=asset)
            except RuntimeError as e:
                logger.debug(e)
                logger.debug(f'Skipping asset "{b}" for "{item.id}".')
                file.remove_p()
                item.assets.pop(b, None)
        try:
            update_item_properties(item, remove_item_props=remove_item_props)
            items.append(item)
        except RuntimeError as e:
            logger.debug(e)
            logger.info(f'Item "{item.id}" is empty, skipping it.')
            item_dir.rmtree_p()
    
    return ItemCollection(items, clone_items=False)

def update_item_properties(x: pystac.Item, remove_item_props=DEFAULT_REMOVE_PROPS):
    """Update item bbox, geometry and proj:epsg introspecting assets.

    Parameters
    ----------
    x : pystac.Item
        The item to update.
    remove_item_props : list of str
        List of regex patterns to remove from item properties.
        If None, no properties are removed.
    
    Returns
    -------
    None
    """

    bbox, geometry, asset_props = properties_from_assets(x.assets)
    x.bbox = bbox
    # as new geometry is a bbox polygon,
    # intersection with old geometry could be more accurate
    geom1 = from_geojson(json.dumps(x.geometry))
    geom2 = from_geojson(json.dumps(geometry))
    geom3 = intersection(geom1, geom2)
    if geom3.is_empty:
        raise RuntimeError("Item geometry is empty.")
    x.geometry = json.loads(to_geojson(geom3))
    x.properties.update(asset_props)

    # remove links
    x.links = []

    if remove_item_props is not None:
        pop_props = []
        for k in x.properties:
            for p in remove_item_props:
                if re.match(p, k):
                    pop_props.append(k)
        for k in pop_props:
            x.properties.pop(k)

def apply_item(x, fun, name, output_dir, overwrite=False,
               copy=True, bbox=None, geometry=None, writer_args=None, **kwargs):
    """
    Applies a function to an item in a collection, 
    saves the result as a raster file and 
    adds the new asset to the item.

    Parameters
    ----------
    x : pystac.Item
        The item to apply the function to.
    fun : function
        The function to apply to the item.
    name : str or list of str
        The name or names of the output raster file(s).
    output_dir : str or list of str
        The directory or directories to save the output raster file(s) to.
    overwrite : bool, optional
        Whether to overwrite existing raster files. Defaults to `False`.
    copy : bool, optional
        Whether to make a copy of the item before applying the function. Defaults to `True`.
        If False, the original item is modified in-place.
    bbox : tuple or None, optional
        The bounding box to clip the raster to. Defaults to `None`.
    geometry : shapely.geometry or None, optional
        The geometry to clip the raster to. Defaults to `None`.
    writer_args : list of dict, optional
        The encoding to use for the raster file. Defaults to `None`.
        See Notes for an example.
    **kwargs : dict
        Additional keyword arguments to pass to the function.

    Returns
    -------
    pystac.Item
        The modified item with the output raster file(s) added as assets.
    
    Notes
    -----

    Example of `writer_args` to encode the two outputs of `fun` in int16 and uint16 respectivelly:

    writer_args=[
        dict(
            encoding=dict(
                dtype="int16", 
                scale_factor=0.001,
                add_offset=0.0,
                _FillValue= 2**15 - 1,
            )
        ),
        dict(
            encoding=dict(
                dtype="uint16", 
                scale_factor= 0.001,
                add_offset= -0.01,
                _FillValue= 2**15 - 1,
            )
        ),
    ]
    """    

# could be a method added to item or collection
    if not isinstance(x, pystac.Item):
        raise ValueError("x must be a pystac.Item")
    
    if copy:
        x = x.clone()


    if isinstance(name, str):
        name = [name]
    if isinstance(output_dir, str):
        output_dir = [output_dir]
    if writer_args is None:
        writer_args = {}
    if isinstance(writer_args, dict):
        writer_args = [writer_args]

    if len(name) != len(output_dir):
        if len(output_dir)==1:
            output_dir = output_dir*len(name)
        else:
            raise ValueError("Argument `output_dir` must have length 1 or the same length as `name`.")

    if len(name) != len(writer_args):
        if len(writer_args)==1:
            writer_args = writer_args*len(name)
        else:
            raise ValueError("Argument `writer_args` must have length 1 or the same length as `name`.")
    
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
            if hasattr(geometry, 'crs') and geometry.crs != arr.rio.crs: # a shapely geometry
                logger.debug(f"Reprojecting geometry from {geometry.crs} to {arr.rio.crs}")
                geometry = geometry.to_crs(arr.rio.crs)
            arr = arr.rio.clip(geometry)
        
        with xr.set_options(keep_attrs=True): 
            res = fun(arr, **kwargs)

        if not isinstance(res, tuple):
            res = (res,)
        if len(res) != Nout:
            raise ValueError(f"Expected {Nout} outputs, got {len(res)}")
        for n,r,f,wa in zip(name, res, raster_file, writer_args):
            if r is None:
                continue
            # write result
            r.name=n
            logger.debug("Writing: ", f)
            write_raster(r, f, overwrite=overwrite, **wa)
    for n, f in zip(name, raster_file):
        if f.exists():
            stac_info = stac_asset_info_from_raster(f)
            asset = pystac.Asset.from_dict(stac_info)    
            x.add_asset(key=n, asset=asset)
    return x

def drop_assets_without_proj(item, inplace=False):
    """
    Drops assets from the given item that do not have the "proj:bbox" field in their extra_fields.

    Parameters:
        item (Item): The item from which to drop assets.
        inplace (bool, optional): If True, the assets will be dropped in place. Otherwise, a clone of the item will be created and modified.

    Returns:
        Item: The modified item with the dropped assets.
    """
    if not inplace:
        item = item.clone()
    item.assets = {k:v for k,v in item.assets.items() if "proj:bbox" in v.extra_fields}
    return item
#######################################

################## Some useful xarray functions ################

def write_raster(x: xr.DataArray, file, driver="COG", overwrite=False, encoding=None, **kwargs):
    """
    Write a raster file from an xarray DataArray.

    Parameters
    ----------
    x : xr.DataArray
        The xarray DataArray to be written as a raster.
    file : str
        The file path to write the raster to.
    driver : str, optional
        The driver to use for writing the raster file. Defaults to "COG".
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. Defaults to False.
        If False, a logger.debug message is printed if the file already exists.
    encoding : dict, optional
        The encoding to use for the raster file. Defaults to None, i.e. float with np.nan as nodata.
    **kwargs
        Additional keyword arguments to be passed to the xarray rio.to_raster() function.

    Returns
    -------
    None

    Notes
    -----
    
    When using encoding scale_factor and add_offset, 
    the dataset `arr` will be "unscaled" as (arr-offset)/scale 
    just before writing it to file.

    Example of encoding:
    encoding=dict(
        dtype="uint16", 
        scale_factor=0.001,
        add_offset=-0.01,
        _FillValue= 2**15 - 1,
    )
    """
    if Path(file).exists() and not overwrite:
        logger.debug(f"File already exists, skipped: {file}")
        return
    if x.dtype == 'bool':
        x = x.astype('uint8')
    if encoding is not None:
        x = x.rio.update_encoding(encoding)
    x.rio.to_raster(file, driver=driver, **kwargs)

def apply_formula(x, formula):
    """Apply formula to bands

    Parameters
    ----------
    x : xarray.DataArray
        It should have a 'band' dimension with the names that will be used by formula.
    formula : str
        Formula, e.g. "B02>700", "CLM > 0", "SLC in [4,5]", "(B08-B06)/(B08+B06)"

    Returns
    -------
    xarray DataArray
        Band operation result
    """
    # formula = "B02 + B03"
    # formula = "CLM in [4,5]"
    bnames = x.band.values.tolist()
    
    for bname in bnames:
        formula = re.sub(f"{bname}", f"x.sel(band='{bname}')", formula)
    
    # replace 'in [...]' by '.isin([...])'
    formula = re.sub(r"\s*in\s*(\[.*\])", ".isin(\\1)", formula)

    return eval(formula)

def harmonize_sen2cor_offet(x, bands=S2_SEN2COR_BANDS, inplace=False):
    """
    Harmonize new Sentinel-2 item collection (Sen2Cor v4+, 2022-01-25)
    to the old baseline (v3-):
    adds an offset of -1000 to the asset extra field "raster:bands" of the items
    with datetime >= 2022-01-25

    Parameters
    ----------
    x: ItemCollection
        An item collection of S2 scenes
    bands: list
        A list of bands to harmonize
    
    inplace: bool
        Whether to modify the collection in place. Defaults to False.
        In that case, a cloned collection is returned.

    Returns
    -------
    ItemCollection
        A collection of S2 scenes with extra_fields["raster:bands"]
        added/updated to each band asset with datetime >= 2022-01-25.
    
    Notes
    -----
    References:
    - https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
    - https://github.com/microsoft/PlanetaryComputer/issues/134
    """
    
    if not inplace:
        x = x.copy()
    for item in x:
        for asset in bands:
            if asset in item.assets:
                if item.properties["datetime"] >= "2022-01-25":
                    item.assets[asset].extra_fields["raster:bands"] = [dict(offset=-1000)]
                else:
                    item.assets[asset].extra_fields["raster:bands"] = [dict(offset=0)]
    if inplace:
        return x

def extract_points(x, df, **kwargs):
    """_summary_

    Parameters
    ----------
    x : xarray.DataArray or xarray.Dataset
    df : pandas.DataFrame
        Coordinates of the points

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The points values, tha can be converted to
        dataframe with `to_dataframe` `to_dask_dataframe`

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import dask.array
    >>> import numpy as np
    >>> da = xr.DataArray(
    ... # np.random.random((100,200)),
    ... dask.array.random.random((100,200,10), chunks=10),
    ... coords = [('x', np.arange(100)+.5), 
    ...           ('y', np.arange(200)+.5),
    ...           ('z', np.arange(10)+.5)]
    ... ).rename("pixel_value")
    >>> df = pd.DataFrame(
    ...    dict(
    ...        x=np.random.permutation(range(100))[:100]+np.random.random(100),
    ...        y=np.random.permutation(range(100))[:100]+np.random.random(100),
    ...        other=range(100),
    ...    )
    ... )
    >>> df.index.rename("id_point", inplace=True)
    >>> extraction = extract_points(da, df, method="nearest", tolerance=.5)
    >>> ext_df = extraction.to_dataframe()
    >>> ext_df.reset_index(drop=False, inplace=True)
    >>> ext_df.rename({k: k+"_pixel" for k in da.dims}, axis=1, inplace=True)
    >>> # join extraction to original dataframe
    >>> df.merge(ext_df, on=["id_point"])

    """
    # x = da
    xk = x.dims
    coords_cols = [c for c in df.keys() if c in xk]
    coords = df[coords_cols]
    points = x.sel(coords.to_xarray(), **kwargs)
    return points
#######################################