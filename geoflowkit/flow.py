import shapely
from shapely.geometry import point
from shapely.errors import EmptyPartError
from shapely.geometry.base import BaseMultipartGeometry

__all__ = ["Flow"]


class Flow(BaseMultipartGeometry):
    """
    A collection of two Points representing origin and destination.

    A Flow represents a movement from an origin point to a destination point.

    Parameters
    ----------
    points : sequence
        A sequence of two Points, or a sequence of (x, y [,z]) numeric coordinate
        pairs or triples, or an array-like of shape (2, 2) or (2, 3).

    Attributes
    ----------
    geoms : sequence
        A sequence of two Points (origin, destination)

    Examples
    --------
    Construct a Flow

    >>> from shapely import Point
    >>> flow = Flow([[0.0, 0.0], [1.0, 2.0]])
    >>> len(flow.geoms)
    2
    >>> flow.o
    <Point POINT (0 0)>
    >>> flow.d
    <Point POINT (1 2)>
    """

    __slots__ = []

    def __new__(self, od_points=None):
        if od_points is None:
            # allow creation of empty Flows, to support unpickling
            return shapely.from_wkt("MULTIPOINT EMPTY")
        elif isinstance(od_points, Flow):
            return od_points
        elif hasattr(od_points, 'coords'):
            # Convert from shapely geometry
            od_points = list(od_points.coords)
        elif hasattr(od_points, '__iter__') and not hasattr(od_points[0], '__len__'):
            # Could be a single coordinate pair, not valid for Flow
            raise ValueError("A Flow must have exactly two points")

        m = len(od_points)
        if m != 2:
            raise ValueError("A Flow must have exactly two points")

        # Create geometry using shapely's multipoints
        subs = [point.Point(od_points[i]) for i in range(m)]
        geom = shapely.multipoints(subs)
        # Change class to Flow so our __str__ is used
        geom.__class__ = Flow
        return geom

    @property
    def __geo_interface__(self):
        return {
            "type": "Flow",
            "coordinates": tuple(g.coords[0] for g in self.geoms),
        }

    def svg(self, scale_factor=1.0, fill_color=None, opacity=None):
        """Returns a group of SVG circle elements for the Flow geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameters.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Default value is 0.6
        """
        if self.is_empty:
            return "<g />"
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            "<g>"
            + "".join(p.svg(scale_factor, fill_color, opacity) for p in self.geoms)
            + "</g>"
        )

    @property
    def wkt(self):
        """Return the Well-Known Text representation of the geometry."""
        if self.is_empty:
            return "FLOW EMPTY"
        coords = ", ".join(f"{p.x} {p.y}" for p in self.geoms)
        return f"FLOW ({coords})"

    @property
    def o(self):
        """Origin point."""
        return self.geoms[0]

    @property
    def d(self):
        """Destination point."""
        return self.geoms[1]

    def __str__(self):
        return self.wkt

    def __repr__(self):
        return self.wkt


# Note: we intentionally do NOT register Flow in shapely.lib.registry
# because Flow is a specialized 2-point MultiPoint, and registering it
# would cause ALL multipoints to become Flow instances.
# Instead, Flow.__new__ uses geom.__class__ = Flow to set the class.


# Patch shapely.to_wkt to use our custom WKT for Flow objects
# This is needed because GeometryArray (geopandas) uses shapely.to_wkt for display
import functools

_original_to_wkt = shapely.to_wkt

@functools.wraps(_original_to_wkt)
def _flow_to_wkt(geom, **kwargs):
    if type(geom) is Flow:  # Use type() check, not isinstance, to avoid matching subclasses
        return geom.wkt
    return _original_to_wkt(geom, **kwargs)

shapely.to_wkt = _flow_to_wkt




