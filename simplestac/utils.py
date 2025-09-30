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
from stac_geoparquet import __version__ as sg_version
import stac_static
from stac_static.search import to_geodataframe
import stackstac
import xarray as xr
import rioxarray # necessary to activate rio plugin in xarray
from tempfile import TemporaryDirectory, NamedTemporaryFile
import time
from tqdm import tqdm
from typing import Union, Iterable
import warnings
import datetime
import geopandas as gpd

from simplestac.local import stac_asset_info_from_raster, properties_from_assets

logger = logging.getLogger(__name__)


#### Generic functions and classes ####
# Adds GDAL_HTTP_MAX_RETRY and GDAL_HTTP_RETRY_DELAY to
# stackstac.rio_reader.DEFAULT_GDAL_ENV
# https://github.com/microsoft/PlanetaryComputerExamples/issues/279
# while waiting for a PR to be merged: https://github.com/gjoseph92/stackstac/pull/232
# See also https://github.com/gjoseph92/stackstac/issues/18
DEFAULT_GDAL_ENV = stackstac.rio_reader.LayeredEnv(
    always=dict(
        GDAL_HTTP_MULTIRANGE="YES",  # unclear if this actually works
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        # ^ unclear if this works either. won't do much when our dask chunks are aligned to the dataset's chunks.
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
    ),
    open=dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        # ^ stop GDAL from requesting `.aux` and `.msk` files from the bucket (speeds up `open` time a lot)
        VSI_CACHE=True
        # ^ cache HTTP requests for opening datasets. This is critical for `ThreadLocalRioDataset`,
        # which re-opens the same URL many times---having the request cached makes subsequent `open`s
        # in different threads snappy.
    ),
    read=dict(
        VSI_CACHE=False
        # ^ *don't* cache HTTP requests for actual data. We don't expect to re-request data,
        # so this would just blow out the HTTP cache that we rely on to make repeated `open`s fast
        # (see above)
    ),
)

S2_THEIA_BANDS = [f"B{i+1}" for i in range(12)]+["B8A"]
S2_SEN2COR_BANDS = [f"B{i+1:02}" for i in range(12)]+["B8A"]

class ExtendPystacClasses:
    """Add capacities to_xarray and filter to pystac Catalog, Collection, ItemCollection"""

    def projv2_to_projv12(self, inplace=False):
        """
        Converts extension projection v2 to projection v1.2, i.e.
        it adds back "proj:epsg" to the asset extra_fields when only "proj:code"
        is present.

        This is necessary while stackstac [issue #262](https://github.com/gjoseph92/stackstac/issues/262)
        is not solved.

        Parameters
        ----------
        inplace : bool
            Whether to modify the collection in place. Defaults to False.

        Returns
        -------
        object
            If `inplace` is False, a cloned item collection is returned.
        """
        if not inplace:
            x = self.clone()
        else:
            x = self

        for item in self.items:
            projv2_to_projv12(item, inplace=True)
        
        if not inplace:
            return x

    def _get_property(self, prop="proj:epsg"):
        prop_list = []
        for item in self:
            p = item.properties
            if prop in p and p[prop]:
                prop_list.append(p[prop])
            else:
                for asset in item.assets.values():
                    pa = asset.extra_fields
                    if prop in pa and pa[prop]:
                        prop_list.append(pa[prop])
        return list(set(prop_list))
    
    def get_epsg(self, unique=True):
        """Get epsg codes from all items and assetsin the collection
        
        Parameters
        ----------
        unique : bool
            If True, a unique epsg code is returned and
            raises an error if multiple epsg codes are found.
            If False, a list of epsg codes is returned.
            Defaults to True.
        
        Returns
        -------
        str or list
        """
        epsg = self._get_property(prop="proj:epsg")
        if unique:
            if len(epsg) == 1:
                epsg = epsg[0]
            else:
                raise ValueError("Multiple epsg found in collection")
        return epsg

    def drop_non_raster(self, pattern="^proj:|^raster:", inplace=False):
        """Drop non raster assets from each item in the collection,
        based on pattern searched in asset extra_fields.
        
        Parameters
        ----------
        pattern : str
            The pattern to search for in asset extra_fields keys.
            Defaults to "^proj:|^raster:".
        inplace : bool
            Whether to modify the collection in place. Defaults to False.
        
        Returns
        -------
        object
            If `inplace` is False, a cloned collection is returned.       
        """
        return self.filter_assets(pattern=pattern, inplace=inplace)
        
    def to_xarray(self, xy_coords="center", bbox=None, geometry=None, gdal_env=DEFAULT_GDAL_ENV, **kwargs):
        """Returns a DASK xarray()
        
        This is a proxy to stackstac.stack

        Parameters
        ----------
        xy_coords : str
            Argument passed to stackstac.stack. Defaults to "center".
        bbox : tuple, list
            A bounding box to clip the xarray to, in the format (xmin, ymin, xmax, ymax).
        geometry : shapely.geometry | geopandas.GeoSeries | geopandas.GeoDataFrame
            A geometry to clip the xarray to.
        gdal_env : stackstac.rio_reader.LayeredEnv
            See stackstac.rio_reader.LayeredEnv. The default is DEFAULT_GDAL_ENV,
            which is the same as stackstac.rio_reader.DEFAULT_GDAL_ENV with the addition
            of GDAL_HTTP_MAX_RETRY=5 and GDAL_HTTP_RETRY_DELAY=1.
        kwargs: dict
            Additional keyword arguments passed to stackstac.stack.
        
        Returns
        -------
        dask.DataArray
            

        Notes
        -----
        The parameters available in stackstac.stack:
        assets=frozenset({'image/jp2', 'image/tiff', 'image/vnd.stac.geotiff', 'image/x.geotiff'}),
        epsg=None, resolution=None, bounds=None, bounds_latlon=None,
        snap_bounds=True, resampling=Resampling.nearest, chunksize=1024,
        dtype=dtype('float64'), fill_value=nan, rescale=True,
        sortby_date='asc', xy_coords='center', properties=True,
        band_coords=True, gdal_env=None,
        errors_as_nodata=(RasterioIOError('HTTP response code: 404'), ),
        reader=<class 'stackstac.rio_reader.AutoParallelRioReader'>

        For details, see [stackstac.stack](https://stackstac.readthedocs.io/en/latest/api/main/stackstac.stack.html).

        Here, xy_coords="center" is the default to be consistent with rioxarray,
        cf https://github.com/gjoseph92/stackstac/issues/207. Otherwise, stackstac.stack has
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
                arr = stackstac.stack(self, xy_coords=xy_coords, gdal_env=gdal_env, bounds=bbox, **kwargs)
            except ValueError as e:
                if "Cannot automatically compute the resolution" in str(e):
                    raise ValueError(str(e)+"\nOr drop non-raster assets from collection with ItemCollection.drop_non_raster()")
                else:
                    raise e
        # it seems that crs is not written correctly by stackstac
        arr = arr.rio.write_crs(arr.rio.crs)
        if bbox is not None:
            arr = arr.rio.clip_box(*bbox)
        if geometry is not None:
            if isinstance(geometry, gpd.GeoDataFrame):
                geometry = geometry.geometry
            if hasattr(geometry, 'crs') and not geometry.crs.equals(arr.rio.crs):
                logger.debug(f"Reprojecting geometry from {geometry.crs} to {arr.rio.crs}")
                geometry = geometry.to_crs(arr.rio.crs)
            arr = arr.rio.clip(geometry)
        return arr
    
    def filter_assets(self, assets=None, pattern=None, drop=False, inplace=False):
        if inplace:
            x = self
        else:
            x = self.clone()
        
        for item in x.items: 
            filter_assets(item, assets=assets, pattern=pattern, drop=drop, inplace=True)

        if not inplace:
            return x

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
                    collection_ready=False,
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
        collection_ready : bool, optional
            If True, the assets directory will be `output_dir / item.id`, ready for a pystac.Collection.
            Defaults to False.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to False.
        inplace : bool, optional
            Whether to modify the collection in place. Defaults to False.
            In that case, a cloned collection is returned.
        datetime : datetime, optional
            A datetime to filter the items with. Defaults to None.
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
                        collection_ready=collection_ready,
                        overwrite=overwrite, copy=False, 
                        bbox=bbox, geometry=geometry,
                        writer_args=writer_args,
                        **kwargs)
        if not inplace:
            return x

    def apply_rolling(self, fun, 
                      name, 
                      output_dir,
                      collection_ready=False,
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
        collection_ready : bool, optional
            If True, the assets directory will be `output_dir / item.id`, ready for a pystac.Collection.
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
            item_id = x.items[i].id

            if center:
                if window%2 == 0:
                    raise ValueError("window must be odd if center=True")
                subitems = x.items[max(i-window//2,0):i+(window-1)//2+1]

            subcol = self.__class__(subitems, clone_items=False)
            raster_dir = output_dir
            if collection_ready:
                raster_dir = [(d / item_id) for d in output_dir]
            raster_file = [d / f"{item_id}_{n}.tif" for n, d in zip(name, raster_dir)]
            

            if not overwrite and all([r.exists() for r in raster_file]):
                logger.debug(f"File already exists, skipping computation: {raster_file}")
                res = tuple([None]*Nout)
            else:
                # compute fun
                with xr.set_options(keep_attrs=True):
                    res = fun(subcol.to_xarray(bbox=bbox, geometry=geometry), **kwargs)
                if res is None:
                    continue
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
                    f.parent.mkdir_p()
                    write_raster(r, f, overwrite=overwrite, **wa)
                    
            for n, f in zip(name, raster_file):
                if f.exists():
                    stac_info = stac_asset_info_from_raster(f)
                    asset = pystac.Asset.from_dict(stac_info)
                    x.items[i].add_asset(key=n, asset=asset)
        
        if not inplace:
            return x

    def extract_points(self, points, method="nearest", tolerance="pixel", drop=False, **kwargs):
        """Extract points from xarray

        Parameters
        ----------
        x : xarray.DataArray or xarray.Dataset
        points : geopandas.GeoDataFrame or pandas.DataFrame
            Points or coordinates of the points
        method, tolerance, drop : see xarray.DataArray.sel
            Additional keyword arguments passed to xarray.DataArray.sel
            If tolerance is "pixel", it is set to half the resolution
            of the xarray, supposing it is a rioxarray.
        **kwargs:
            Additional keyword arguments passed to `ItemCollection.to_xarray()`
            
        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The points values with points index as coordinate.
            The returned xarray can then be converted to
            dataframe with `to_dataframe` or `to_dask_dataframe`.

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

        # avoid starting anything if not all points
        if isinstance(points, (gpd.GeoDataFrame, gpd.GeoSeries)):
            if not points.geom_type.isin(['Point', 'MultiPoint']).all():
                raise ValueError("All geometries must be of type Point or MultiPoint")
        
        arr = self.to_xarray(**kwargs)#geometry=points)
        if tolerance == "pixel":
            tolerance = arr.rio.resolution()[0] / 2
        return extract_points(arr, points, method=method, tolerance=tolerance, drop=drop)

class ItemCollection(pystac.ItemCollection, ExtendPystacClasses):
    def __init__(self, items: Iterable[ItemLike], **kwargs):
        super().__init__(items, **kwargs)
        if sg_version > "0.3.2":
            unify_properties(self, inplace=True)
        if stackstac.__version__ <= "0.5.1":
            self.projv2_to_projv12(inplace=True)

# class Catalog(pystac.Catalog, ExtendPystacClasses):
#     pass

# class Collection(pystac.Collection, ExtendPystacClasses):
#     pass

DEFAULT_REMOVE_PROPS = ['.*percentage', 'eo:cloud_cover', '.*mean_solar.*']

def write_assets(x: Union[ItemCollection, pystac.Item],
                 output_dir: str,
                 bbox=None,
                 geometry=None,
                 keep_asset_attrs=True,
                 xy_coords='center', 
                 remove_item_props=DEFAULT_REMOVE_PROPS,
                 overwrite=False,
                 progress=True,
                 writer_args=None,
                 inplace=False,
                 modifier=None,
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
        Argument forwarded to ItemCollection.to_xarray.
        The bounding box (in the CRS of the items) to clip the assets to.
    geometry : Optional
        Argument forwarded to ItemCollection.to_xarray to rioxarray.clip the assets to.
        Usually a GeoDataFrame or GeoSeries.
        See notes.
    keep_asset_attrs : bool, optional
        Whether to keep the asset attributes in the returned item collection.
        Defaults to True.
    xy_coords : str, optional
        Argument forwarded to ItemCollection.to_xarray.
    remove_item_props : list of str
        List of regex patterns to remove from item properties.
        If None, no properties are removed.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    writer_args : dict, optional
        Arguments to pass to write_raster for each asset. Defaults to `None`.
        See Notes for an example.
    inplace : bool, optional
        Whether to modify the input collection in place or clone it. Defaults to False (i.e. clone).
    modifier : function, optional
        A callable that modifies the children collection and items
        returned by this Client. This can be useful for injecting
        authentication parameters into child assets to access data
        from non-public sources, see pystac_client.Client for details.
    **kwargs
        Additional keyword arguments passed to write_raster.

    Returns
    -------
    ItemCollection
        The item collection with the metadata updated with local asset paths.
    
    Notes
    -----
    Arguments `bbox` and `geometry` are to ways to clip the assets before writing.
    Although they look similar, they may lead to different results. 
    First, `bbox` does not have CRS, thus it is to the user to know
    in which CRS x.to_xarray() will be before being clipped. If geometry is used instead,
    it is automatically converted to the collection xarray CRS.
    Second, as we use the default arguments for rio.clip and rio.clip_box,
    the clip_box with bbox will contain all touched pixels while the clip with geometry will
    contain only pixels whose center is within the polygon (all_touched=False).
    Adding a buffer of resolution/2 could be a workaround to avoid that,
    i.e. keep all touched pixels while clipping with a geometry.

    The `writer_args` argument can be used to specify the writing arguments (e.g. encoding) for specific assets.
    Thus, it must be a dictionary with the keys corresponding to asset keys.
    If the asset key is not in `writer_args`, the `kwargs` are passed to `write_raster`.
    The following example would encode the B02 band as int16, and the rest of the assets as float:
    writer_args = {
        "B02": {
            "encoding": {
                "dtype": "int16",
            }
        }
    }

    """    
    if isinstance(x, pystac.Item):
        x = ItemCollection([x])
    
    if not inplace:
        x = x.clone()

    output_dir = Path(output_dir).expand()
    items = []
    for item in tqdm(x, disable=not progress):
        ic = ItemCollection([item], clone_items=True)
        if modifier is not None:
            ic = modifier(ic)
        arr = ic.to_xarray(bbox=bbox, geometry=geometry,xy_coords=xy_coords, ).squeeze("time")
        item_dir = (output_dir / item.id).mkdir_p()
        for b in arr.band.values:
            filename = '_'.join([item.id, b+'.tif'])
            file = item_dir / f"{filename}"
            
            # Use specific writer args if available
            if writer_args is not None and b in writer_args:
                wa = writer_args[b]
            else:
                wa = kwargs
            try:
                if file.exists() and not overwrite:
                    logger.info(f"File already exists, skipping asset: {file}")
                else:
                    done = False
                    max_retry = 10
                    retry = 0
                    wait = 2
                    while not done and retry != max_retry:
                        try:
                            write_raster(arr.sel(band=b), file, **wa)
                            done=True
                        except RuntimeError as e:
                            logger.info(e)
                            if 'HTTP response code: 403' in str(e):
                                retry += 1
                                logger.info(
                                    f"Failed to read the asset '{b}' of item '{item.id}', "
                                    f"retrying in {wait*retry} minutes ({retry}/{max_retry}).")
                                time.sleep(wait*60)
                            else:
                                raise e

                    if not file.exists():
                        raise Exception(f"File was not written: {file}")
                    
                # update stac asset info            
                stac_info = stac_asset_info_from_raster(file)
                if keep_asset_attrs:
                    asset_info = item.assets[b].to_dict()
                    asset_info.update(stac_info)
                    stac_info = asset_info
                asset = pystac.Asset.from_dict(stac_info)
                item.add_asset(key=b, asset=asset)
            except RuntimeError as e:
                logger.info(e)
                logger.info(f"There was an error writing the asset '{b}' of item '{item.id}', skipping it.")
                file.remove_p()
                item.assets.pop(b, None)
        
        update_item_properties(item, remove_item_props=remove_item_props)
        items.append(item)
    
    if not inplace:
        return x

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

def unify_properties(x: ItemCollection, method="fill",inplace=False):
    """
    Fill or Drop property fields not shared
    by all items.

    Some ItemCollections may not have the same item property fields
    for all items. As it can be an issue for generating a dataframe
    with stac-geoparquet > 0.3.2,
    cf [issue #76](https://github.com/stac-utils/stac-geoparquet/issues/76),
    this function fills or drops the missing fields depending on the method.

    Parameters
    ----------
    x : ItemCollection
        The ItemCollection to unify properties.
    method : str
        Either "fill" or "drop".
    inplace : bool
        If True, the collection is modified in place.

    Returns
    -------
    ItemCollection if inplace is False, else None.
    """
    if not inplace:
        x = x.copy()

    keys = set(x.items[0].properties.keys())
    udiff = []
    if method == "fill":
        for item in x.items:
            keys = keys.union(item.properties.keys())
        
        for item in x.items:
            diff = keys.symmetric_difference(item.properties.keys())
            for k in diff:
                item.properties[k]=None
                udiff.append(k)
        if len(udiff) > 0:
            print(f"Filled property fields with None when missing in items:\n{set(udiff)}")
    elif method == "drop":
        for item in x.items:
            keys = keys.intersection(item.properties.keys())
            
        for item in x.items:
            diff = keys.symmetric_difference(item.properties.keys())
            for k in diff:
                item.properties.pop(k)
                udiff.append(k)
        
        if len(udiff) > 0:
            print(f"Dropped property fields that were not shared by all items:\n{set(udiff)}")
    
    else:
        raise ValueError(f"Unknown method: {method}")

    if not inplace:
        return x

def apply_item(x, fun, name, output_dir, collection_ready=False, overwrite=False,
               copy=True, bbox=None, geometry=None, writer_args=None, 
               xarray_args=dict(xy_coords="center"), **kwargs):
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
    collection_ready : bool, optional
        If True, the assets directory will be `output_dir / item.id`, ready for a pystac.Collection.
        Defaults to `False`.
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
    xarray_args : dict, optional
        The arguments to pass to `stackstac.stack`. Defaults to `dict(xy_coords="center")`.
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
                _FillValue=-2**15,
            )
        ),
        dict(
            encoding=dict(
                dtype="uint16", 
                scale_factor=0.001,
                add_offset=-0.01,
                _FillValue=2**15 - 1,
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
    
    # add item id level: output_dir / item.id
    if collection_ready:
        output_dir = [(d / x.id).mkdir_p() for d in output_dir]

    raster_file = [d / f"{x.id}_{n}.tif" for n, d in zip(name, output_dir)]
    if not overwrite and all([r.exists() for r in raster_file]):
        logger.debug(f"File already exists, skipping computation: {raster_file}")
        res = tuple([None]*Nout)
    else:
        # compute fun
        if xarray_args is None:
            xarray_args = {}

        arr = ItemCollection([x]).to_xarray(bbox=bbox, geometry=geometry, **xarray_args)
    
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

    # add assets to item
    for n, f in zip(name, raster_file):
        if f.exists():
            stac_info = stac_asset_info_from_raster(f)
            asset = pystac.Asset.from_dict(stac_info)    
            x.add_asset(key=n, asset=asset)
    return x

def drop_assets_without_proj(item, pattern="^proj:|^raster:", inplace=False):
    """
    Drops assets from the given item that do not have 
    extra_fields with "proj:" or "raster:" prefix.

    Parameters
    ----------
    item: pystac.Item
      The item from which to drop assets.
    pattern: str, optional. 
        The pattern to search for in asset extra_fields keys.
    inplace: bool, optional
        If True, the assets will be dropped in place.
        Otherwise, a clone of the item will be created and modified.

    Returns
    ------
    pystac.Item
        The modified item with the dropped assets.
    """
    if not inplace:
        item = item.clone()
    
    item.assets = {k:v for k,v in item.assets.items() if any([bool(re.search(pattern, p)) for p in v.extra_fields])}
    
    if len(item.assets) == 0:
        logger.warning(f"Item {item.id} has no raster assets.")

    return item

def filter_assets(
        item: pystac.Item,
        assets: Union[str, list]=None,
        pattern: str="^proj:|^raster:",
        drop: bool=False,
        inplace: bool=False):
    """
    Filter assets from the given item according to pattern and asset keys.

    Parameters
    ----------
    item: pystac.Item
      The item from which to filter assets.
    assets: Union[str, list], optional
        The asset keys to match.
    pattern: str, optional. 
        The pattern to search for in asset extra_fields keys.
    drop: bool, optional
        If True, the assets matching the pattern and the asset keys
        are dropped.
    inplace: bool, optional
        If True, the assets will be filtered in place.
        Otherwise, a clone of the item will be created and modified.

    Returns
    ------
    pystac.Item
        The modified item.
    """
    if not inplace:
        item = item.clone()
    
    if not pattern:
        keep = item.assets.keys()
    else:
        keep = []
        for k,v in item.assets.items():
            if any([bool(re.search(pattern, p)) for p in v.extra_fields]):
                keep.append(k)
    
    if assets is not None:
        if not isinstance(assets, list):
            assets = [assets]
        keep = [k for k in keep if k in assets]

    if drop:
        item.assets = {k:v for k,v in item.assets.items() if k not in keep}
    else:
        item.assets = {k:v for k,v in item.assets.items() if k in keep}

    if len(item.assets) == 0:
        logger.warning(f"Item {item.id} has no assets left after filtering.")

    return item

def projv2_to_projv12(item: pystac.Item, inplace=False):
    """
    Converts extension projection v2 to projection v1.2, i.e.
    it adds back "proj:epsg" to the asset extra_fields when only "proj:code"
    is present.

    This is necessary while stackstac [issue #262](https://github.com/gjoseph92/stackstac/issues/262)
    is not solved.

    Parameters
    ----------
    item: pystac.Item
      The item from which to filter assets.
    inplace: bool, optional
        If True, the assets will be filtered in place.
        Otherwise, a clone of the item will be created and modified.

    Returns
    ------
    pystac.Item
        The modified item.
    """
    if not inplace:
        item = item.clone()

    schema_v2 = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
    schema_v1_2 = "https://stac-extensions.github.io/projection/v1.2.0/schema.json"
    if schema_v2 in item.stac_extensions:
        item.stac_extensions.remove(schema_v2)
        item.stac_extensions.append(schema_v1_2)
        
        for k,v in item.assets.items():
            if "proj:epsg" not in v.extra_fields and "proj:code" in v.extra_fields:
                if v.extra_fields["proj:code"].startswith("EPSG:"):
                    v.extra_fields["proj:epsg"] = int(re.sub("EPSG:", "", v.extra_fields["proj:code"]))
    
    return item

def harmonize_sen2cor_offset(x: ItemCollection, assets=S2_SEN2COR_BANDS, inplace=False):
    """
    Harmonize new Sentinel-2 item with Sen2Cor v4+.

    Since Sen2Cor v4.0.0 (2022-01-25), an offset of -1000 has been added to the
    reflectance values of the spectral bands in order to have them in
    uint16 instead of int16. This function adds an offset of -1000 to
    the spectral band assets extra field "raster:bands" for items with
    processing baseline >= v4.
    
    The processing baseline version is expected in properties
    processing:version or s2:processing_baseline with format
    "xx.yy", e.g. "05.10".

    Parameters
    ----------
    x: ItemCollection
        An item collection of S2 scenes
    
    assets: list
        A list of assets (i.e. bands) to harmonize
            
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

    Warning, if None
    """
    offsetv4=-1000
    proc_prop = "processing:version"
    s2_prop = "s2:processing_baseline"
    
    if proc_prop in x[0].properties:
        prop = proc_prop
    elif s2_prop in x[0].properties:
        prop = s2_prop
    else:
        raise ValueError("No processing:version or s2:processing_baseline found in item collection.")

    if not inplace:
        x = x.clone()
    for item in x:
        for k in assets:
            if k in item.assets:
                asset = item.assets[k]
                # add raster:bands if not already there
                if not "raster:bands" in asset.extra_fields:
                    asset.extra_fields["raster:bands"] = [dict(offset=0)]
                # update raster:bands offset for specific dates
                rb = asset.extra_fields["raster:bands"][0]
                if float(item.properties[prop]) >= 4.0:
                    scale = rb["scale"] if "scale" in rb else 1.
                    rb.update(dict(offset=offsetv4*scale))
    if not inplace:
        return x

def update_scale_offset(x, scale=None, offset=None, rescale_offset=True, assets=S2_SEN2COR_BANDS, inplace=False):
    """Updates/add scale and offset to STAC asset extra_fields["raster:bands"].

    When scale and offset are defined, they are used in ItemCollection.to_xarray().
    In that case, reflectance = scale*asset + offset, with reflectance the xarray value.
    **Warning**: nothing is changed in the asset file itself, it just impacts the
    way the ItemCollection is converted to xarray.

    Parameters
    ----------
    x : ItemCollection
        ItemCollection to be modified
    scale : float, optional
        The scale to apply to the asset extra_fields["raster:bands"], by default None.
    offset : float, optional
        The offset to apply to the asset extra_fields["raster:bands"], by default None.
    rescale_offset : bool, optional
        Whether to rescale already existing offset, by default True.
        If offset is not None, offset is not rescaled.
    assets : list, optional
        List of assets to update, by default S2_SEN2COR_BANDS
    inplace : bool, optional
        Whether to modify the collection in place, by default False.
    
    Returns
    -------
    ItemCollection
        The modified ItemCollection, or None if inplace=True.
    """
    if not inplace:
        x = x.clone()
    for item in x:
        for k in assets:
            if k in item.assets:
                asset = item.assets[k]
                if not "raster:bands" in asset.extra_fields:
                    # add only one element list: considering only one band
                    asset.extra_fields["raster:bands"] = [dict()]
                rb = asset.extra_fields["raster:bands"][0]
                if scale is not None:
                    old_scale = rb.get("scale", 1.)
                    rb.update(dict(scale=scale))
                    if rescale_offset and "offset" in rb:
                        rb["offset"] = rb["offset"] * scale / old_scale
                if offset is not None:
                    rb.update(dict(offset=offset))

    if not inplace:
        return x
#######################################

################## Some useful xarray functions ################

def write_raster(
        x: xr.DataArray, file: str, driver="GTIFF", compress="DEFLATE",
        tiled=True, overwrite=False, encoding=None, compute=True, **kwargs):
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
    compute : bool, optional
        If True and data is a dask array, then compute and save the data immediately.
        If False, return a dask Delayed object. Default is True.
        See rioxarray docs for details.
    **kwargs
        Additional keyword arguments to be passed to the xarray rio.to_raster() function.

    Returns
    -------
    None or dask.Delayed

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
    if compute:
        with TemporaryDirectory(dir=file.parent) as tmpdir:
            tmpfile = Path(tmpdir) / Path(file).name
            res = x.rio.to_raster(tmpfile, driver=driver,
                                compress=compress, tiled=tiled,
                                **kwargs)
            tmpfile.move(file)
    else:
        res = x.rio.to_raster(file, driver=driver,
                            compress=compress, tiled=tiled,
                            compute=False, **kwargs)
    return res

def apply_formula(x, formula):
    """Apply formula to bands

    Parameters
    ----------
    x : xarray.DataArray
        It should have a 'band' dimension with the names that will be used by formula.
    formula : str
        Formula, e.g. "B02 > 600", "CLM > 0", "B02 > 600 | ~SLC in [4,5]", "(B08-B06)/(B08+B06)"

    Returns
    -------
    xarray DataArray
        Band operation result
    """
    # formula = "B02 + B03"
    # formula = "CLM in [4,5]"
    if isinstance(x, xr.DataArray):
        bnames = x.band.values.tolist()
        
        for bname in bnames:
            formula = re.sub(f"{bname}", f"x.sel(band='{bname}')", formula)
    elif isinstance(x, xr.Dataset):
        bnames = list(x)
        for bname in bnames:
            formula = re.sub(f"{bname}", f"x['{bname}']", formula)
    else:
        raise TypeError("x should be xarray.DataArray or xarray.Dataset")

    # replace 'in [...]' by '.isin([...])'
    formula = re.sub(r"\s*in\s*(\[.*\])", ".isin(\\1)", formula)
    

    return eval(formula)

def extract_points(x, points, method=None, tolerance=None, drop=False):
    """Extract points from xarray

    Parameters
    ----------
    x : xarray.DataArray or xarray.Dataset
    points : geopandas.GeoDataFrame or pandas.DataFrame
        Points or coordinates of the points
    method, tolerance, drop : see xarray.DataArray.sel
        Additional keyword arguments passed to xarray.DataArray.sel

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The points values with points index as coordinate.
        The returned xarray can then be converted to
        dataframe with `to_dataframe` or `to_dask_dataframe`.

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
    valid_types = (gpd.GeoDataFrame, gpd.GeoSeries)
    if isinstance(points, valid_types):
        if not points.geom_type.isin(['Point', 'MultiPoint']).all():
            raise ValueError("All geometries must be of type Point")

    if isinstance(points, valid_types):
        if hasattr(points, 'crs') and not points.crs.equals(x.rio.crs):
            logger.debug(f"Reprojecting points from {points.crs} to {x.rio.crs}")
            points = points.to_crs(x.rio.crs)
        points = points.get_coordinates()

    xk = x.dims
    coords_cols = [c for c in points.keys() if c in xk]
    coords = points[coords_cols]
    points = x.sel(coords.to_xarray(), method=method, tolerance=tolerance, drop=drop)
    return points

def add_reduced_coords(da, da1, dim):
    """Add reduced coords to xarray

    When reducing xarray along a dimension,
    it drops all coordinates linked to that dimension.
    This function adds them back if they have unique
    values along that dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The reference dataarray.
    da1 : xarray.DataArray
        The reduced dataarray.
    dim : str
        The reduced dimension.

    Returns
    -------
    xarray.DataArray
        The dataarray with added coords.
    
    Notes
    -----
    See issue https://github.com/pydata/xarray/issues/8317
        
    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> N=6
    >>> da = xr.DataArray(np.arange(1.*N**3).reshape(N,N,N), dims=["x", "y", "time"], 
    ...               coords={
    ...                   "x": np.arange(N),
    ...                   "y": np.arange(N),
    ...                   "time": [pd.NaT]*N,
    ...                   "start": ("time", [1.]*3 + [3.]*3),
    ...                   "end": ("time", ["2."]*2 + ["2."]+ ["4."]*3),
    ...                   "tile":("time", ["16PFS", "16PGS", "16PKS"]*2),
    ...                   "autre": "autre",})
    >>> da = da.set_xindex(["start", "tile"]).unstack("time")
    >>> da1 = da.mean("time")
    >>> add_reduced_coords(da, da1, "time")
    """
    for k,c in da.coords.items():
        if len(c.dims) > 0 and dim in c.dims:
            axis = np.where(np.array(c.dims)==dim)[0]
            # unique coordinate values along axis
            uc = np.unique(c, axis=axis).squeeze()
            # new coordinate dims
            nd = list(c.dims)
            nd.remove(dim)
            # try to add it, if it fails we just ignore it
            try:
                da1 = da1.assign_coords({k:(nd, uc)})
            except:
                pass
    return da1
################################################################