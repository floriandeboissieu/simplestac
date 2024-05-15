# v1.1.2
- fix issue of drop_non_raster with no proj:bbox: now looking for any "proj:" or "raster:" properties

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
