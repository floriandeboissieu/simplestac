"""
Deal with STAC Extents.

# AutoSpatialExtent

The vanilla `pystac.SpatialExtent` enables to describe a spatial extent 
from several bounding boxes. While this is useful, sometimes we want to
merge together bounding boxes that are partially overlapping. For instance,
we can take the example of the France mainland, covered by multiple remote
sensing products that are generally partially overlapping, and the Corse 
island that is also covered by a number of RS products, but spatially 
disjoint from the France mainland RS products bounding boxes. In this 
particular exemple, we would like to regroup all RS products bounding 
boxes so that there is one bbox for the France mainland, and another 
bbox for the Corse island. This is particularly useful when a STAC 
collection covers sparsely a broad area (e.g. worldwide), with several 
isolated regions.

The `AutoSpatialExtent` is an extension of the `pystac.SpatialExtent`.

Instances are initialized with the same arguments as `pystac.SpatialExtent`.
Internally, bounding boxes lists are processed at initialisation, so all 
partially overlapping bounding boxes are merged and updated as a single one.

# AutoTemporalExtent

The `AutoTemporalExtent` is an extension of the `pystac.TemporalExtent`.
It computes the date min and date max of a set of dates or dates ranges.

"""
import pystac
from dataclasses import dataclass
from datetime import datetime
from typing import Union
from stacflow.common_types import Bbox


@dataclass
class SmartBbox:
    """
    Small class to work with a single 2D bounding box.
    """
    coords: Bbox = None  # [xmin, ymin, xmax, ymax]

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


def clusterize_bboxes(bboxes: list[Bbox]) -> list[Bbox]:
    """
    Computes a list of bounding boxes regrouping all overlapping ones.

    Args:
        bboxes: 2D bounding boxes (list of int of float)

    Returns:
        list of 2D bounding boxes (list of int of float)

    """
    bboxes = [SmartBbox(bbox) for bbox in bboxes]
    clusters = [bboxes.pop()]

    while bboxes:
        bbox = bboxes.pop()
        inter_clusters = [
            i for i, cluster in enumerate(clusters) if bbox.touches(cluster)
        ]
        if inter_clusters:
            # We merge all intersecting clusters into one
            clusters[inter_clusters[0]].update(bbox)
            for i in inter_clusters[1:]:
                clusters[inter_clusters[0]].update(clusters[i])
            clusters = [
                cluster
                for i, cluster in enumerate(clusters)
                if i not in inter_clusters[1:]
            ]
        else:
            clusters.append(bbox)

    return [cluster.coords for cluster in clusters]


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

    def update(self, other: pystac.SpatialExtent | Bbox):
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
                all_dates += [i for i in interval if i is not None]
            elif isinstance(interval, datetime):
                all_dates.append(interval)
            else:
                TypeError(f"Unsupported date/range of: {interval}")
        self.intervals = \
            [[min(all_dates), max(all_dates)]] if all_dates else [None, None]
