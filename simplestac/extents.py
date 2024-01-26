"""
Module to deal with STAC Extents.
"""
import pystac
from dataclasses import dataclass
from datetime import datetime
from typing import Union

Coords = list[Union[int, float]]


@dataclass
class SmartBbox:
    """
    Small class to work with a single 2D bounding box.
    """
    coords: Coords = None  # [xmin, ymin, xmax, ymax]

    def touches(self, other: "SmartBbox") -> bool:
        """
        Overlap test.

        Args:
            other: other bounding box

        Returns:
            True if the other bounding box touches, else False.

        """
        xmin, ymin, xmax, ymax = self.coords
        o_xmin, o_ymin, o_xmax, o_ymax = other.coords

        if xmax < o_xmin or o_xmax < xmin or ymax < o_ymin or o_ymax < ymin:
            return False
        return True

    def update(self, other: "SmartBbox"):
        """
        Update the coordinates of the Bbox. Modifies itself inplace.

        Args:
            other: other bounding box

        """
        if not self.coords:
            self.coords = other.coords
        else:
            self.coords = [
                min(self.coords[0], other.coords[0]),
                min(self.coords[1], other.coords[1]),
                max(self.coords[2], other.coords[2]),
                max(self.coords[3], other.coords[3])
            ]


def clusterize_bboxes(bboxes: list[Coords]) -> list[Coords]:
    """
    Computes a list of bounding boxes regrouping all overlapping ones.

    Args:
        bboxes: 2D bounding boxes (list of int of float)

    Returns:
        list of 2D bounding boxes (list of int of float)

    """
    # Regroup bboxes into clusters of bboxes
    smart_bboxes = [SmartBbox(bbox) for bbox in bboxes]
    clusters = bboxes_to_bboxes_clusters(smart_bboxes)

    # Compute clusters extents
    clusters_unions = [SmartBbox() for _ in clusters]
    for i, cluster in enumerate(clusters):
        for smart_bbox in cluster:
            clusters_unions[i].update(smart_bbox)

    return [smart_bbox.coords.copy() for smart_bbox in clusters_unions]


def bboxes_to_bboxes_clusters(smart_bboxes: list[SmartBbox]) -> list[
    list[SmartBbox]]:
    """
    Transform a list of bounding boxes into a nested list of clustered ones.

    Args:
        smart_bboxes: a list of `SmartBbox` instances

    Returns:
        a list of `SmartBbox` instances list

    """
    clusters_labels = compute_smart_bboxes_clusters(smart_bboxes)
    clusters_bboxes = [[] for _ in range(max(clusters_labels) + 1)]
    for smart_bbox, labels in zip(smart_bboxes, clusters_labels):
        clusters_bboxes[labels].append(smart_bbox)
    return clusters_bboxes


def compute_smart_bboxes_clusters(smart_bboxes: list[SmartBbox]) -> list[int]:
    """
    Compute the extent of a cluster of `SmartBbox` instances.

    Args:
        smart_bboxes: a list of `SmartBbox` instances

    Returns:
        a vector of same size as `smart_bboxes` with the group numbers (int)

    """
    labels = len(smart_bboxes) * [None]
    group = 0

    def dfs(index: int):
        """
        Deep first search with o(n) complexity.

        Args:
            index: vertex index.

        """
        labels[index] = group
        cur_item = smart_bboxes[index]
        for i, (item, label) in enumerate(zip(smart_bboxes, labels)):
            if i != index and cur_item.touches(item) and labels[i] is None:
                dfs(i)

    while any(label is None for label in labels):
        next_unmarked = next(
            i for i, label in enumerate(labels)
            if label is None
        )
        dfs(next_unmarked)
        group += 1
    return labels


class AutoSpatialExtent(pystac.SpatialExtent):
    """
    Custom extension of pystac.SpatialExtent that automatically compute bboxes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializer. Clusterize boxes after the original initializer.

        Args:
            *args: args
            **kwargs: keyword args

        """
        super().__init__(*args, **kwargs)
        self.clusterize_bboxes()

    def update(self, other: pystac.SpatialExtent | Coords):
        """
        Updates itself with a new spatial extent or bounding box. Modifies
        inplace `self.bboxes`.

        Args:
            other: spatial extent or bbox coordinates

        """
        is_spat_ext = isinstance(other, pystac.SpatialExtent)
        self.bboxes += other.bboxes if is_spat_ext else other
        self.clusterize_bboxes()

    def clusterize_bboxes(self):
        """
        Regroup the bounding boxes that overlap. Modifies inplace `self.bboxes`.

        """
        self.bboxes = clusterize_bboxes(self.bboxes)


class AutoTemporalExtent(pystac.TemporalExtent):
    """
    Custom extension of pystac.TemporalExtent that automatically updates itself
    with another date or temporal extent provided.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializer. Regroup all intervals into a single one.

        Args:
            *args: args
            **kwargs: keyword args

        """
        super().__init__(*args, **kwargs)
        self.make_single_interval()

    def update(self, other: Union[pystac.TemporalExtent, datetime]):
        """
        Updates itself with a new temporal extent of date. Modifies inplace
        `self.intervals`.

        Args:
            other: temporal extent or datetime

        """
        is_temp_ext = isinstance(other, pystac.TemporalExtent)
        intervals = other.intervals if is_temp_ext else [[other, other]]
        self.intervals += intervals
        self.make_single_interval()

    def make_single_interval(self):
        all_dates = []
        for interval in self.intervals:
            if isinstance(interval, (list, tuple)):
                all_dates += [i for i in interval]
            elif isinstance(interval, datetime):
                all_dates.append(interval)
            else:
                TypeError(f"Unsupported date/range of: {interval}")
        self.intervals = [[min(all_dates), max(all_dates)]]
