"""This module aims at providing utilities to work with STAC ItemCollection.
"""

import logging
import numpy as np
from path import Path
import pandas as pd
import pystac
from pystac.item_collection import ItemLike
import stac_static
from stac_static.search import to_geodataframe
import stackstac
import xarray as xr
from tqdm import tqdm
import warnings

from simplestac.local import stac_asset_info_from_raster

logger = logging.getLogger(__name__)


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
                    stac_info = stac_asset_info_from_raster(f)
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
            stac_info = stac_asset_info_from_raster(f)
            asset = pystac.Asset.from_dict(stac_info)    
            x.add_asset(key=n, asset=asset)
    return x

#######################################

