# Flow Metrics Module Design

## Context

The geoflowkit package currently has `flowprocess.py` containing only `pairwise_distances()`. As new flow metrics are needed (k-order nearest neighbor distance, SNN distance, flow entropy, flow divergence), there's an opportunity to reorganize into a dedicated `flowmetrics.py` module that consolidates all flow-to-flow measurement functions.

## Decision

- **Rename** `flowprocess.py` → `flowmetrics.py`
- Consolidate all distance/imilarity metrics in this module
- Ensure backward compatibility: update all imports in `clustering/` and other modules

## Architecture

### Module: `geoflowkit/flowmetrics.py`

**Existing functions (from flowprocess.py):**
- `pairwise_distances(fdf, distance='max', ...)` — pairwise flow distance matrix

**New functions:**

1. **`k_neighbor_distances(fdf, k, distance='max', dis_matrix=None)`**
   - Returns array of shape (n_flows,) — each flow's k-order nearest neighbor distance
   - Internally reuses `pairwise_distances()` if `dis_matrix` not provided
   - Helper: `_k_neighbor_distances_from_matrix(dis_matrix, k)`

2. **`snn_distance(fdf, k, distance='max')`**
   - Returns (n_flows, n_flows) SNN similarity matrix
   - Based on Eq. 2-14: combines origin and destination KNN intersections
   - Helper: `_knn_neighborhood(coords, k)` — returns KNN neighbor indices for origin/destination points
   - Helper: `_snn_from_knn(knn_o, knn_d, k)` — computes SNN from KNN indices

3. **`flow_entropy(fdf, cell_area=None)`**
   - Returns scalar — overall flow space entropy (Eq. 2-17)
   - Uses flow count per OD zone pair
   - Optional `cell_area` for spatial entropy weighting

4. **`flow_divergence(fdf, n_directions=6)`**
   - Returns scalar — flow directional entropy (Eq. 2-18)
   - Bins flows by angle into `n_directions` sectors
   - Helper: `_angle_to_bin(angle, n_directions)`

### Module: `geoflowkit/clustering/dbscan.py` & `kmedoid.py`

Update imports:
```python
from geoflowkit.flowmetrics import pairwise_distances
```

### Module: `geoflowkit/spatial/kl_function.py`

Update imports similarly.

## Verification

1. Run existing tests: `pytest tests/ -k "flowprocess or pairwise"`
2. Add new tests for each metric function
3. Verify clustering modules still work after import changes

## Files to Modify

- `geoflowkit/flowprocess.py` → rename to `geoflowkit/flowmetrics.py`
- `geoflowkit/clustering/dbscan.py` — update import
- `geoflowkit/clustering/kmedoid.py` — update import
- `geoflowkit/spatial/kl_function.py` — update import
- `geoflowkit/__init__.py` — add `flowmetrics` export
- `tests/` — add test file for flowmetrics
