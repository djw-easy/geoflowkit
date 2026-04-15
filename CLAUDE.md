# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoFlowKit is a Python package for handling and analyzing geographical flow data (origin-destination pairs), extending pandas and geopandas with flow-specific operations.

## Common Commands

```bash
# Install package in development mode
pip install -e .

# Run tests
pytest tests/

# Run a specific test
pytest tests/test_flow_series.py -k test_name

# Build package for distribution
python -m build

# Publish to PyPI (triggered by git tags matching v*)
git tag v0.1.0 && git push origin v0.1.0
```

## Architecture

### Core Class Hierarchy

```
shapely.BaseMultipartGeometry
    └── Flow (geoflowkit/flow.py)
        - 2-point MultiPoint geometry representing origin-destination pair
        - Properties: o (origin point), d (destination point)

pandas.Series + GeoPandasBase
    └── FlowSeries (geoflowkit/flowseries.py)
        - Stores Flow objects in a Series
        - Inherits spatial operations from GeoPandasBase

geopandas.GeoDataFrame
    └── FlowDataFrame (geoflowkit/flowdataframe.py)
        - DataFrame with Flow geometry column
        - Supports flow attributes and spatial operations

geoflowkit.base.FlowBase (mixin)
    └── Shared properties: o, d, length, angle, volume, density, within(), clip(), distance()
```

### Module Structure

```
geoflowkit/
├── flow.py              # Flow geometry class (shapely extension)
├── flowseries.py        # FlowSeries (pandas Series subclass)
├── flowdataframe.py     # FlowDataFrame (geopandas GeoDataFrame subclass)
├── base.py              # FlowBase mixin with shared flow properties
├── io.py                # read_csv, read_file, flows_from_od, flows_from_geometry
├── flowmetrics.py       # pairwise_distances, k_neighbor_distances, snn_distance, flow_entropy, flow_divergence
├── spatial/             # Spatial heterogeneity of geographical flow
│   ├── utils.py         # nth_largest, second_order_density
│   ├── kl_function.py   # k_func, l_func, local_l_func
│   └── centrality.py    # i_index
├── clustering/          # Flow clustering
│   ├── kmedoid.py       # KMedoidFlow, kmedoid()
│   └── dbscan.py        # DBSCANFlow, dbscan()
├── manifold/            # FTSNE for dimensionality reduction
│   └── ftsne/
│       ├── ftsne.py     # Main FTSNE class
│       └── utils.py     # Gradient descent, KL divergence, probability calculations
```

### Key Design Patterns

**Flow Geometry**: Flow extends shapely's `BaseMultipartGeometry` as a specialized 2-point MultiPoint. It's intentionally NOT registered in shapely's geometry registry to avoid all multipoints becoming Flow instances. The class is set via `geom.__class__ = Flow` after construction.

**CRS Handling**: Geographic CRS operations warn users about potential issues. Use `to_crs()` to project data before spatial operations. The `check_geographic_crs()` method in FlowBase warns when planar operations are performed on geographic coordinates.

**I/O**: CSV reading uses custom logic to parse origin/destination columns. GeoPackage/GeoJSON reading delegates to geopandas.

## Data Model

### Flow Representation
- Origin `o` and destination `d` are stored as the two points in a Flow geometry
- Derived properties: `length` (distance between o and d), `angle` (direction in radians)
- Flow DataFrame can contain additional attributes (value, category, etc.)

### Flow Metrics
- `pairwise_distances(fdf, distance='max')` computes flow-to-flow distances
- Distance methods: 'max' (default), 'sum', 'min', 'mean', 'weighted'
- Weighted distance incorporates flow lengths when `length=True`
- `k_neighbor_distances(fdf, k)` returns k-order nearest neighbor distances for each flow
- `snn_distance(fdf, k)` computes Shared Nearest Neighbor distance (based on origin/destination KNN intersections)
- `flow_entropy(fdf, cell_area=None)` calculates flow space entropy
- `flow_divergence(fdf, n_directions=6)` calculates flow directional entropy

## Notable Implementation Details

- FTSNE uses sklearn's parameter validation via `@validate_params` decorator
- Clustering algorithms (kmedoid, dbscan) use flow-specific distance metrics
- The `within()` method checks both origin AND destination points against a mask
- FlowSeries.plot() and FlowDataFrame.plot(kind='arrow') render flows as arrows using matplotlib quiver and support a zoom parameter to control the view extent; FlowDataFrame.plot() additionally supports categorical columns for multi-color plotting
