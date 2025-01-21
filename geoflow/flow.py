import shapely
from shapely.errors import EmptyPartError
from shapely.geometry import point
from shapely.geometry.base import BaseMultipartGeometry

__all__ = ["Flow"]


class Flow(BaseMultipartGeometry):
    """
    A collection of one or more Points.

    A Flow has zero area and zero length.

    Parameters
    ----------
    points : sequence
        A sequence of Points, or a sequence of (x, y [,z]) numeric coordinate
        pairs or triples, or an array-like of shape (N, 2) or (N, 3).

    Attributes
    ----------
    geoms : sequence
        A sequence of Points

    Examples
    --------
    Construct a Flow containing two Points

    >>> from shapely import Point
    >>> ob = Flow([[0.0, 0.0], [1.0, 2.0]])
    >>> len(ob.geoms)
    2
    >>> type(ob.geoms[0]) == Point
    True
    """

    __slots__ = []

    def __new__(self, points=None):
        if points is None:
            # allow creation of empty Flows, to support unpickling
            # TODO better empty constructor
            return shapely.from_wkt("Flow EMPTY")
        elif isinstance(points, Flow):
            return points

        m = len(points)
        assert m == 2, "A Flow must have exactly two points"
        subs = []
        for i in range(m):
            p = point.Point(points[i])
            if p.is_empty:
                raise EmptyPartError("Can't create Flow with empty component")
            subs.append(p)

        if len(points) == 0:
            return shapely.from_wkt("Flow EMPTY")

        return shapely.multipoints(subs)

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


shapely.lib.registry[4] = Flow


if __name__ == "__main__":
    flow = Flow([[0, 0], [1, 1]])
    print(flow, type(flow))

