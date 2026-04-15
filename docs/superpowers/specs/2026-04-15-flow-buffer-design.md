# Flow Buffer Analysis — Design Spec

## Overview

Implement flow buffer analysis for GeoFlowKit, following Section 4.2 of the flow analysis literature. A flow's buffer is the region in flow-space within distance R of a given flow.

## Background

For a flow `f_c` with origin `O_c` and destination `D_c`, and radius R:

- **Max distance** (Eq 4-14): A flow `f` is in the buffer if `max(dist(O_c, O), dist(D_c, D)) <= R`
  - Geometrically: O must be within circle `Circle(O_c, R)` AND D must be within `Circle(D_c, R)`
  - The buffer region in 2D projection is the **intersection** of the two circles → a lens/almond shape

- **Sum distance**: R^o + R^d = R, radii proportional to flow's origin/destination distances
  - Buffer region = intersection of two circles with potentially different radii

## API Design

### `FlowBase.buffer()` — shared by Flow and FlowDataFrame

```python
Flow.buffer(radius: float, distance: str = 'max') -> Polygon
FlowDataFrame.buffer(radius: float, distance: str = 'max') -> GeoDataFrame[Polygon]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | float | — | Buffer radius R |
| `distance` | str | `'max'` | `'max'` (Eq 4-14, fixed R) or `'sum'` (R^o + R^d = R) |

### Returns

- `Flow.buffer()` → `shapely.Polygon`: Single flow's buffer geometry (intersection of two circles)
- `FlowDataFrame.buffer()` → `geopandas.GeoDataFrame`: One row per flow, geometry = Polygon

### Geometry

Each flow's buffer = `shapely.intersection(circle_o, circle_d)`:
- `circle_o = Point(O).buffer(R)` or `Point(O).buffer(R^o)` for sum distance
- `circle_d = Point(D).buffer(R)` or `Point(D).buffer(R^d)` for sum distance
- Result is a lens/almond-shaped Polygon

### CRS Handling

Geographic CRS triggers a warning (consistent with existing `check_geographic_crs()` pattern). User should project data before buffering.

## File Structure

```
geoflowkit/
├── base.py         # FlowBase.buffer() method
```

## Implementation Notes

- `FlowDataFrame.buffer()` uses `apply()` to call `Flow.buffer()` on each row
- For `'sum'` distance: compute `R^o = R * dist(O_c, O) / (dist(O_c, O) + dist(D_c, D))` and `R^d = R - R^o`
- Shapely's `Point.buffer(r)` produces a Polygon circle with default 16 segments (sufficient for visualization)
- Return new GeoDataFrame with `crs` preserved from input
- Buffer overlay operations (union/intersection of multiple flow buffers) are separate future steps — not in scope here

## Scope

- Only Euclidean (planar) buffer — geographic/haversine not in scope
- Only `max` and `sum` distance modes
- Single `Flow.buffer()` returns Polygon; no multi-polygon output per flow
