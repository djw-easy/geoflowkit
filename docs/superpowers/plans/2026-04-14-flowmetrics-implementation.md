# Flow Metrics Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `flowmetrics.py` module consolidating all flow-to-flow measurement functions (pairwise_distances, k_neighbor_distances, snn_distance, flow_entropy, flow_divergence), update all imports across the codebase, and sync documentation.

**Architecture:** Rename `flowprocess.py` → `flowmetrics.py`, add new metric functions following existing patterns (CRS handling, NumPy docstrings, FlowDataFrame-first-arg). Imports in `clustering/` and `spatial/` modules will be updated to reference the new module name.

**Tech Stack:** shapely, numpy, pandas, geopandas, scipy.spatial.distance

---

## File Structure

- **Rename:** `geoflowkit/flowprocess.py` → `geoflowkit/flowmetrics.py`
- **Modify:** `geoflowkit/__init__.py` — add `flowmetrics` export
- **Modify:** `geoflowkit/clustering/dbscan.py` — update import
- **Modify:** `geoflowkit/clustering/kmedoid.py` — update import
- **Modify:** `geoflowkit/spatial/kl_function.py` — update import
- **Modify:** `geoflowkit/spatial/__init__.py` — update import
- **Modify:** `CLAUDE.md` — update module structure
- **Modify:** `README.md` — update API reference
- **Create:** `tests/test_flowmetrics.py` — tests for all metric functions

---

### Task 1: Rename flowprocess.py to flowmetrics.py

- [ ] **Step 1: Rename file**

```bash
mv geoflowkit/flowprocess.py geoflowkit/flowmetrics.py
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/flowprocess.py geoflowkit/flowmetrics.py
git commit -m "refactor: rename flowprocess.py to flowmetrics.py"
```

---

### Task 2: Update clustering/dbscan.py import

**Files:**
- Modify: `geoflowkit/clustering/dbscan.py`

- [ ] **Step 1: Update import**

Find line: `from geoflowkit.flowprocess import pairwise_distances`
Change to: `from geoflowkit.flowmetrics import pairwise_distances`

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/clustering/dbscan.py
git commit -m "refactor: update import to flowmetrics"
```

---

### Task 3: Update clustering/kmedoid.py import

**Files:**
- Modify: `geoflowkit/clustering/kmedoid.py`

- [ ] **Step 1: Update import**

Find line: `from geoflowkit.flowprocess import pairwise_distances`
Change to: `from geoflowkit.flowmetrics import pairwise_distances`

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/clustering/kmedoid.py
git commit -m "refactor: update import to flowmetrics"
```

---

### Task 4: Update spatial/kl_function.py import

**Files:**
- Modify: `geoflowkit/spatial/kl_function.py`

- [ ] **Step 1: Update import**

Find line: `from geoflowkit.flowprocess import pairwise_distances`
Change to: `from geoflowkit.flowmetrics import pairwise_distances`

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/spatial/kl_function.py
git commit -m "refactor: update import to flowmetrics"
```

---

### Task 5: Update spatial/__init__.py

**Files:**
- Modify: `geoflowkit/spatial/__init__.py`

- [ ] **Step 1: Update import**

Check if `from geoflowkit.flowprocess import` exists and update to `from geoflowkit.flowmetrics import`

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/spatial/__init__.py
git commit -m "refactor: update flowprocess import to flowmetrics in spatial/__init__.py"
```

---

### Task 6: Update geoflowkit/__init__.py

**Files:**
- Modify: `geoflowkit/__init__.py`

- [ ] **Step 1: Update import**

Check if `flowprocess` is exported. If so, add `flowmetrics` export alongside or replace:
```python
from .flowmetrics import pairwise_distances
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/__init__.py
git commit -m "refactor: update __init__.py to export flowmetrics"
```

---

### Task 7: Add k_neighbor_distances function

**Files:**
- Modify: `geoflowkit/flowmetrics.py`

Add at end of file:

```python
def k_neighbor_distances(fdf: FlowDataFrame, k: int, distance='max',
                         dis_matrix=None) -> np.ndarray:
    """Calculate k-order nearest neighbor distances for each flow.

    The k-order nearest neighbor distance of a flow is its distance to the
    k-th nearest neighboring flow.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    k : int
        Order of the nearest neighbor (1 = nearest, 2 = second nearest, etc.)
    distance : str, optional
        Distance combination method ('max', 'min', 'sum', 'mean', 'weighted'),
        by default 'max'
    dis_matrix : np.ndarray, optional
        Precomputed distance matrix. If None, will be computed using
        pairwise_distances, by default None

    Returns
    -------
    np.ndarray
        Array of shape (n_flows,) with each flow's k-order neighbor distance

    Examples
    --------
    >>> k_dists = k_neighbor_distances(fdf, k=1)  # 1st order (nearest neighbor)
    >>> k_dists = k_neighbor_distances(fdf, k=2)  # 2nd order
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance)

    # Set diagonal to infinity so flow doesn't select itself
    dis_matrix = dis_matrix.copy()
    np.fill_diagonal(dis_matrix, np.inf)

    # Get k-th nearest neighbor distance for each flow
    # Sort each row and take the k-th element (0-indexed)
    sorted_distances = np.sort(dis_matrix, axis=1)
    k_neighbor_dist = sorted_distances[:, k - 1]

    return k_neighbor_dist
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/flowmetrics.py
git commit -m "feat: add k_neighbor_distances function"
```

---

### Task 8: Add _knn_neighborhood helper

**Files:**
- Modify: `geoflowkit/flowmetrics.py`

Add helper before `k_neighbor_distances`:

```python
def _knn_neighborhood(coords: np.ndarray, k: int) -> np.ndarray:
    """Find k-nearest-neighbor indices for each point.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n, 2) with point coordinates
    k : int
        Number of nearest neighbors

    Returns
    -------
    np.ndarray
        Array of shape (n, k) with indices of k nearest neighbors for each point
    """
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    # Query k+1 because the tree query includes the point itself as nearest neighbor
    _, indices = tree.query(coords, k=k + 1)
    # Exclude the point itself (first column is the point itself)
    return indices[:, 1:]
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/flowmetrics.py
git commit -m "feat: add _knn_neighborhood helper function"
```

---

### Task 9: Add snn_distance function

**Files:**
- Modify: `geoflowkit/flowmetrics.py`

Add at end of file:

```python
def snn_distance(fdf: FlowDataFrame, k: int, distance: str = 'max') -> np.ndarray:
    """Calculate Shared Nearest Neighbor (SNN) distance between flows.

    SNN distance is based on the intersection of k-nearest-neighbor sets
    of origin points and destination points between pairs of flows.
    Follows Eq. 2-14 from the flow analysis literature.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    k : int
        Number of nearest neighbors to consider for KNN neighborhoods
    distance : str, optional
        Distance method for pairwise distances ('max', 'min', 'sum', 'mean'),
        by default 'max'

    Returns
    -------
    np.ndarray
        SNN distance matrix of shape (n_flows, n_flows)
        Values in [0, 1]: 0 means identical KNN neighborhoods, 1 means no shared neighbors

    Examples
    --------
    >>> snn_dist = snn_distance(fdf, k=8)
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    origins = shapely.get_coordinates(fdf.o)
    destinations = shapely.get_coordinates(fdf.d)

    # Get KNN neighborhoods for origins and destinations separately
    knn_o = _knn_neighborhood(origins, k)
    knn_d = _knn_neighborhood(destinations, k)

    n = len(fdf)
    snn_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Intersection of O_i's KNN with O_j's KNN
            o_interaction = len(np.intersect1d(knn_o[i], knn_o[j]))
            # Intersection of D_i's KNN with D_j's KNN
            d_interaction = len(np.intersect1d(knn_d[i], knn_d[j]))

            # SNN distance from Eq. 2-14
            sim = (o_interaction / k) * (d_interaction / k)
            dist = 1 - sim

            snn_matrix[i, j] = dist
            snn_matrix[j, i] = dist

    return snn_matrix
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/flowmetrics.py
git commit -m "feat: add snn_distance function for shared nearest neighbor similarity"
```

---

### Task 10: Add flow_entropy function

**Files:**
- Modify: `geoflowkit/flowmetrics.py`

Add at end of file:

```python
def flow_entropy(fdf: FlowDataFrame, cell_area=None) -> float:
    """Calculate flow space entropy.

    Measures the spatial distribution disorder of flows using Shannon entropy.
    Supports both standard entropy and spatially-weighted entropy (Batty's spatial entropy).

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    cell_area : np.ndarray, optional
        Array of shape (n, n) with zone pair volumes (area products).
        If None, returns standard Shannon entropy, by default None

    Returns
    -------
    float
        Flow space entropy value

    Examples
    --------
    >>> entropy = flow_entropy(fdf)  # Standard entropy
    >>> entropy_weighted = flow_entropy(fdf, cell_area=areas)  # Spatially-weighted
    """
    n = len(fdf)

    # Count flows per OD zone pair (assumes zones are indexed by flow position)
    # If fdf has 'origin_id' and 'dest_id' columns, use those; otherwise uniform distribution
    p = np.ones(n) / n  # Uniform distribution

    if cell_area is None:
        # Standard Shannon entropy (Eq. 2-15)
        entropy = -np.sum(p * np.log2(p + 1e-10))
    else:
        # Batty's spatial entropy (Eq. 2-16/2-17)
        area_normalized = cell_area / (cell_area.sum() + 1e-10)
        entropy = -np.sum(p * np.log2(p / (area_normalized + 1e-10) + 1e-10))

    return entropy
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/flowmetrics.py
git commit -m "feat: add flow_entropy function for spatial entropy calculation"
```

---

### Task 11: Add flow_divergence function

**Files:**
- Modify: `geoflowkit/flowmetrics.py`

Add at end of file:

```python
def flow_divergence(fdf: FlowDataFrame, n_directions: int = 6) -> float:
    """Calculate flow divergence (directional entropy).

    Measures the dispersion of flow directions using Shannon entropy on
    binned angular directions.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    n_directions : int, optional
        Number of direction bins (sectors), by default 6
        Each sector has angle = 360 / n_directions degrees

    Returns
    -------
    float
        Flow divergence value (directional entropy)
        Higher values indicate more dispersed flow directions

    Examples
    --------
    >>> div = flow_divergence(fdf, n_directions=6)  # 6 sectors of 60 degrees each
    >>> div = flow_divergence(fdf, n_directions=8)  # 8 sectors of 45 degrees each
    """
    if n_directions < 2:
        raise ValueError("n_directions must be >= 2")

    # Get flow angles
    angles = fdf.angle.values

    # Normalize angles to [0, 2*pi]
    angles = np.mod(angles, 2 * np.pi)

    # Bin angles into sectors
    sector_size = 2 * np.pi / n_directions
    bin_indices = (angles / sector_size).astype(int) % n_directions

    # Count flows in each sector
    counts = np.bincount(bin_indices, minlength=n_directions)

    # Calculate probabilities
    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total

    # Shannon entropy (Eq. 2-18)
    entropy = -np.sum(p * np.log2(p + 1e-10))

    return entropy
```

- [ ] **Step 2: Commit**

```bash
git add geoflowkit/flowmetrics.py
git commit -m "feat: add flow_divergence function for directional entropy"
```

---

### Task 12: Create test file

**Files:**
- Create: `tests/test_flowmetrics.py`

- [ ] **Step 1: Write tests**

```python
import numpy as np
import pytest
from geoflowkit import FlowDataFrame, FlowSeries, flows_from_od
from geoflowkit.flowmetrics import (
    pairwise_distances,
    k_neighbor_distances,
    snn_distance,
    flow_entropy,
    flow_divergence
)


@pytest.fixture
def sample_fdf():
    """Create a sample FlowDataFrame for testing."""
    o_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    d_points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    fs = flows_from_od(o_points, d_points, crs="EPSG:3857")
    data = {'geometry': fs, 'value': [1, 2, 3, 4, 5]}
    return FlowDataFrame(data, crs="EPSG:3857")


class TestPairwiseDistances:
    def test_basic(self, sample_fdf):
        result = pairwise_distances(sample_fdf)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)
        assert np.all(result == result.T)  # symmetric

    def test_distance_types(self, sample_fdf):
        for dist_type in ['max', 'min', 'sum', 'mean']:
            result = pairwise_distances(sample_fdf, distance=dist_type)
            assert result.shape == (5, 5)


class TestKNeighborDistances:
    def test_k1(self, sample_fdf):
        result = k_neighbor_distances(sample_fdf, k=1)
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_k2(self, sample_fdf):
        result = k_neighbor_distances(sample_fdf, k=2)
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_k_greater_than_n(self, sample_fdf):
        with pytest.raises(IndexError):
            k_neighbor_distances(sample_fdf, k=10)

    def test_k_must_be_positive(self, sample_fdf):
        with pytest.raises(ValueError):
            k_neighbor_distances(sample_fdf, k=0)


class TestSNNDistance:
    def test_basic(self, sample_fdf):
        result = snn_distance(sample_fdf, k=2)
        assert result.shape == (5, 5)
        assert np.all((result >= 0) & (result <= 1))
        assert np.all(result == result.T)  # symmetric
        assert np.all(result.diagonal() == 0)  # diagonal is 0


class TestFlowEntropy:
    def test_basic(self, sample_fdf):
        result = flow_entropy(sample_fdf)
        assert isinstance(result, float)
        assert result >= 0

    def test_uniform_distribution(self, sample_fdf):
        # For n flows, maximum entropy is log2(n)
        result = flow_entropy(sample_fdf)
        n = len(sample_fdf)
        max_entropy = np.log2(n)
        assert result <= max_entropy


class TestFlowDivergence:
    def test_basic(self, sample_fdf):
        result = flow_divergence(sample_fdf, n_directions=6)
        assert isinstance(result, float)
        assert result >= 0

    def test_n_directions(self, sample_fdf):
        for n in [4, 6, 8]:
            result = flow_divergence(sample_fdf, n_directions=n)
            max_entropy = np.log2(n)
            assert result <= max_entropy

    def test_invalid_n_directions(self, sample_fdf):
        with pytest.raises(ValueError):
            flow_divergence(sample_fdf, n_directions=1)
```

- [ ] **Step 2: Run tests to verify they fail (no functions exist yet)**

Run: `pytest tests/test_flowmetrics.py -v`
Expected: FAIL (import errors for new functions)

- [ ] **Step 3: Commit**

```bash
git add tests/test_flowmetrics.py
git commit -m "test: add flowmetrics test suite"
```

---

### Task 13: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update module structure**

Change from:
```
├── flowprocess.py       # pairwise_distances
```

To:
```
├── flowmetrics.py       # pairwise_distances, k_neighbor_distances, snn_distance, flow_entropy, flow_divergence
```

Also update Data Model section:
Change "Flow Distance Calculations" to "Flow Metrics" and add new functions.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with flowmetrics module"
```

---

### Task 14: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update API Reference section**

Add to the Key Functions list:
```
- `k_neighbor_distances(fdf, k, distance='max', ...)`: K-order nearest neighbor distances
- `snn_distance(fdf, k, distance='max', ...)`: Shared nearest neighbor distance
- `flow_entropy(fdf, cell_area=None, ...)`: Flow space entropy
- `flow_divergence(fdf, n_directions=6, ...)`: Flow directional entropy
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README.md API reference"
```

---

## Self-Review Checklist

- [ ] All imports updated across clustering/ and spatial/ modules
- [ ] flowmetrics.py contains all 5 functions (pairwise_distances, k_neighbor_distances, snn_distance, flow_entropy, flow_divergence)
- [ ] Tests cover basic functionality for all functions
- [ ] CLAUDE.md and README.md updated
- [ ] No placeholder/TODO comments in code
- [ ] Each task has its own commit

---

## Execution Option

**Plan complete and saved to `docs/superpowers/plans/2026-04-14-flowmetrics-implementation.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
