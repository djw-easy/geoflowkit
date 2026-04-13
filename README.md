# GeoFlowKit

[![PyPI version](https://badge.fury.io/py/geoflowkit.svg)](https://badge.fury.io/py/geoflowkit)

A Python package for handling and analyzing geographical flow data, extending pandas and geopandas with flow-specific operations.

## Overview

GeoFlowKit provides `FlowSeries` and `FlowDataFrame` types, which are subclasses of `pandas.Series` and `pandas.DataFrame` respectively. They are designed to work with flow data consisting of origin-destination (OD) pairs, similar to how `geopandas.GeoSeries` and `geopandas.GeoDataFrame` work with geometries.

## Installation

```bash
pip install geoflowkit
```

Or install from source:

```bash
pip install .
```

### Dependencies

- shapely
- numpy
- pandas
- geopandas >= 1.0.1
- matplotlib
- scikit-learn
- tqdm
- numba

## Quick Start

### Creating Flow Objects

```python
import numpy as np
from geoflowkit import Flow, FlowSeries, FlowDataFrame

# Create a single Flow (origin-destination pair)
flow = Flow([[0, 0], [1, 1]])

# Access origin and destination points
print(flow.o)  # POINT (0 0)
print(flow.d)  # POINT (1 1)
```

### Creating FlowSeries

```python
# From a list of Flow objects
fs = FlowSeries([
    Flow([[0, 0], [1, 1]]),
    Flow([[1, 1], [2, 2]]),
    Flow([[2, 2], [3, 3]])
], crs="EPSG:4326")

# From coordinate arrays using flows_from_od
from geoflowkit import flows_from_od

o_points = np.array([[0, 0], [1, 1], [2, 2]])
d_points = np.array([[1, 1], [2, 2], [3, 3]])
fs = flows_from_od(o_points, d_points, crs="EPSG:4326")
```

### Creating FlowDataFrame

```python
# Create a FlowDataFrame with attributes
data = {
    'id': [1, 2, 3],
    'value': [10, 20, 30],
    'geometry': fs
}
fdf = FlowDataFrame(data, crs="EPSG:4326")
print(fdf)
```

### Reading Data from Files

```python
# Read from CSV (specify origin/destination columns)
fdf = read_csv(
    'flow_data.csv',
    use_cols=['ox', 'oy', 'dx', 'dy'],
    crs='EPSG:4326'
)

# Read from GeoPackage
fdf = read_file('flow_data.gpkg', layer='flows')
```

## Core Features

### Flow Properties

```python
# Access origin and destination points
origins = fdf.o  # GeoSeries of origin points
destinations = fdf.d  # GeoSeries of destination points

# Flow length and angle
lengths = fdf.length  # Distance from origin to destination
angles = fdf.angle    # Direction of flow (radians)

# Flow density and volume
density = fdf.density  # Flows per unit area
volume = fdf.volume    # Total bounding area
```

### Spatial Operations

```python
# Calculate pairwise distances between flows
from geoflowkit import pairwise_distances

dist_matrix = pairwise_distances(fdf, distance='max')

# Clip flows within a polygon
clipped = fdf.clip(polygon_mask)

# Select flows within bounds
within_bounds = fdf.within(bounds_box)
```

### Spatial Clustering Scale Detection (K/L Functions)

```python
# Calculate K function for spatial clustering
from geoflowkit import k_func, l_func

r_list, kr_list = k_func(fdf, dr=0.1, k=1)
r_list, lr_list = l_func(fdf, dr=0.1, k=1)

# Local L function for individual flows
from geoflowkit import local_l_func

llrs = local_l_func(fdf, r=0.5)
```

### Grid Aggregation

```python
# Divide study area into grid and aggregate flows
gridded = fdf.to_grid(delta_x=0.1, delta_y=0.1)
```

### Visualization

```python
# Plot flows as arrows
ax = fdf.plot(kind='arrow', column='value')

# Plot FlowSeries
ax = fs.plot()
```

### Flow Clustering

```python
# K-medoid clustering
from geoflowkit import kmedoid

labels = kmedoid(fdf, n_clusters=5)

# DBSCAN clustering
from geoflowkit import dbscan

labels = dbscan(fdf, eps=0.5, min_samples=5)
```

### Manifold Learning (FTSNE)

```python
from geoflowkit import FTSNE

# Global interpretability (separate O and D)
transformer = FTSNE(perplexity=200, learning_rate='auto')
X_embedded = transformer.fit_transform(
    fdf,
    identity={'o': 0, 'd': 1}
)

# Local interpretability (union O and D)
X_embedded = transformer.fit_transform(
    fdf,
    union={('o', 'd'): (0, 1)}
)
```

## Examples

Jupyter notebook examples are available in the `examples/` folder:

- [basic_usage.ipynb](examples/basic_usage.ipynb) - Basic usage of Flow, FlowSeries, and FlowDataFrame
- [clustering.ipynb](examples/clustering.ipynb) - K-medoid clustering for flow data
- [kl_function.ipynb](examples/kl_function.ipynb) - K/L functions for spatial clustering detection
- [ft_sne.ipynb](examples/ft_sne.ipynb) - FTSNE manifold learning for flow data

## API Reference

### Classes

- `Flow`: Geometry object representing an origin-destination pair
- `FlowSeries`: pandas Series subclass for storing Flow objects
- `FlowDataFrame`: pandas DataFrame subclass with Flow geometry column

### Key Functions

- `flows_from_od(o, d, crs=None)`: Create FlowSeries from coordinate arrays
- `flows_from_geometry(geometry, crs=None)`: Create FlowSeries from geometry objects
- `read_csv(file_path, use_cols, crs=None, **kwargs)`: Read flow data from CSV
- `read_file(file_path, **kwargs)`: Read flow data from vector file
- `pairwise_distances(fdf, distance='max', ...)`: Calculate flow distance matrix
- `k_func(fdf, dr, k=1, distance='max', ...)`: K function for spatial clustering detection
- `l_func(fdf, dr, k=1, distance='max', ...)`: L function for spatial clustering detection
- `local_l_func(fdf, r, distance='max', ...)`: Local L function for individual flows
- `kmedoid(fdf, n_clusters=5, ...)`: K-medoid clustering for flows
- `dbscan(fdf, eps=0.5, min_samples=5, ...)`: DBSCAN clustering for flows

## License

GeoFlowKit is licensed under the MIT License.

## Contact

For questions or feedback: djw@lreis.ac.cn
