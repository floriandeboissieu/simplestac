import pytest
from simplestac.extents import AutoSpatialExtent

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
