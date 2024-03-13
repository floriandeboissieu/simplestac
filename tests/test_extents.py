import pytest
from simplestac.extents import AutoSpatialExtent

from datetime import datetime

def test_spatial_extent():
    """
    Test the `AutoSpatialExtent` class.

    Two clusters of bboxes (i.e. lists of bboxes) composed respectively with 2 and 1 bbox are 
    created (by definition, the clusters are disjoint: their bboxes don't overlap)
    We instanciate an `AutoSpatialExtent` and we check that the two expected clusters are found.

    """
    # first cluster (e.g. "france mainland")
    bbox1 = [4, 42, 6, 44]
    bbox2 = [3, 41, 5, 43]

    # second cluster (e.g. "corse")
    bbox3 = [7, 42, 8, 50]

    ase = AutoSpatialExtent(bboxes=[bbox1, bbox2, bbox3])
    assert ase.bboxes == [[7, 42, 8, 50], [3, 41, 6, 44]]

def test_temporal_extent():
    """
    Test the `AutoTemporalExtent`.

    """
    # dates only (as plain list)
    dates = [
        datetime(year=2020, month=1, day=1),
        datetime(year=2022, month=1, day=1),
        datetime(year=2023, month=1, day=1),
    ]
    auto_text = AutoTemporalExtent(dates)
    assert auto_text.intervals == [[
        datetime(year=2020, month=1, day=1),
        datetime(year=2023, month=1, day=1)
    ]]

    # dates only (as nested list)
    auto_text = AutoTemporalExtent([dates])
    assert auto_text.intervals == [[
        datetime(year=2020, month=1, day=1),
        datetime(year=2023, month=1, day=1)
    ]]

    # mixed dates + ranges
    auto_text = AutoTemporalExtent([
        datetime(year=2020, month=1, day=1),
        [None, None],
        [datetime(year=2019, month=1, day=1), None],
        [None, datetime(year=2024, month=1, day=1)],
        datetime(year=2023, month=1, day=1)
    ])
    assert auto_text.intervals == [[
        datetime(year=2019, month=1, day=1),
        datetime(year=2024, month=1, day=1)
    ]]
