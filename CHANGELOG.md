
# v1.2.5

## Fix
- `harmonize_sen2cor_offset` a double if condition prevented to use some sensor
  versions (MR !16).
- `projv2_to_projv12` for the case of `ItemCollection.from_file` (issue #13)
- if none of the expected assets are present in an item directory: add a warning
  and skips item (issue #11)
- changed pattern for THEIA format to include S2C and to avoid item pattern
  match with sub-directories (e.g. {pat}_PVD_ALL, ...) (issue #11)

# v1.2.4
## Add
- support for zip files when building ItemCollection (issue #8)

## Change
- remove constraint on pystac version

# v1.2.3
## Add
- `ItemCollection.get_epsg`: get the epsg of items or assets

## Fix
- `unify_properties` for `inplace=False`
- `harmonize_sen2cor_offset` based on processing baseline version instead of
  datetime (issue #5)
- `update_scale_offset` unscale offset before rescale if new scale (issue #6)

# v1.2.2
## Add
- `ItemCollection.filter_assets`: filter assets (keep or drop)

## Fix
- log as info if writing had an error (issue #2)
- convert projection v2 to projection v1.2 (issue #3)

# v1.2.1
## Add
- add xarray.Dataset support to apply_formula
- make write_raster ready for delayed write

# v1.2.0
## Change
- changed default write_raster args from driver="COG" to {driver="GTIFF", compress="DEFLATE", tiled=True},
  as COG has no benefit for local use (overviews takes processing time and disk space).
  --> increases writing speed by x2-3 as overviews do not have to be computed
- made the COG validation quiet at asset creation

## Fix
- CRS issue in ItemCollection.to_xarray

# v1.1.3

## Add
- function `update_scale_offset` to add or update raster scale and offset values to the assets of the collection. These values will then be automatically used to deliver rescaled values when calling to_xarray.

## Fix
- issue with new version of stac-geoparquet, cf https://github.com/stac-utils/stac-geoparquet/issues/76
- numpy < 2.0 in environment.yml (issue with numpy >= 2.0)

# v1.1.2

## Add
- parameter `pattern` to `ItemCollection.drop_non_raster` and `drop_asset_without_proj`
- support for recursive item search in `build_item_collection`
- parameter `collection_ready` to `apply_item`, `apply_items`, `apply_rolling`

## Fix
- fix issue of drop_non_raster with no proj:bbox: now looking for any "proj:" or "raster:" properties.
  A parameter `pattern` was added to 

# v1.1.1

## Add
- add modifier to write_assets
- add function add_reduced_coords to fix the issue https://github.com/pydata/xarray/issues/8317

## Fix
- end_datetime expanding by default to the end of the day in seconds, e.g. 2019-12-31T23:59:59Z.

# v1.1.0

## Add

- function `write_assets`: write item assets (rasters only at the moment) of an ItemCollection locally and return the corresponding ItemCollection.
- function `harmonize_sen2cor_offset`: adds an `offset` property to the assets so it is taken into account by `to_xarray`.
- method `ItemCollection.drop_duplicates`: drop duplicated ID returned by pgstac.
- method `ItemCollection.drop_non_raster`: drop non raster assets.
- method `ItemCollection.to_xarray` default arg `gdal_env`: now it inlcudes GDAL retries in case of failure while reading url
- function `extract_points` and method `ItemCollection.extract_points` to extract points time series.
- `writer_args` to `ItemCollection.apply_...` methods and function in order to specify the outputs format, e.g. the encoding.
- in local.py, `start_datetime` and `end_datetime` can now be used instead of `datetime` in the template used to build a local ItemCollection.
- module `extents.py` to manipulate STAC extents.
- tests for CI

## Fix

- `apply_formula` with "in" operator in apply_formula.
- COG type of local STAC assets (instead of GTiff)
- imports in `simplestac.utils`
