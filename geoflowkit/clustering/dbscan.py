"""DBSCAN clustering for FlowDataFrame objects using flow-specific distance metrics."""

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from geoflowkit import FlowDataFrame


def _flow_distance_factory(distance: str = 'max'):
    """Create a flow-specific distance metric function.

    Parameters
    ----------
    distance : str, default='max'
        Distance type: 'max', 'min', 'sum', 'mean'.
        - 'max': Maximum of origin and destination distances
        - 'min': Minimum of origin and destination distances
        - 'sum': Sum of origin and destination distances
        - 'mean': Average of origin and destination distances

    Returns
    -------
    callable
        A metric function with signature (u, v) -> float
        where u, v are 1D arrays of shape (4,) containing
        [o_x, o_y, d_x, d_y].

    Raises
    ------
    ValueError
        If an invalid distance type is specified.
    """
    if distance not in {'max', 'min', 'sum', 'mean'}:
        raise ValueError(
            f"distance must be one of 'max', 'min', 'sum', 'mean', "
            f"got {distance}"
        )

    def flow_metric(u, v):
        """Compute distance between two flows.

        Parameters
        ----------
        u : array-like, shape (4,)
            Origin and destination coordinates [o_x, o_y, d_x, d_y]
        v : array-like, shape (4,)
            Origin and destination coordinates [o_x, o_y, d_x, d_y]

        Returns
        -------
        float
            The flow distance between u and v.
        """
        # Extract origin and destination coordinates
        o_u, d_u = u[:2], u[2:]
        o_v, d_v = v[:2], v[2:]

        # Compute origin and destination distances using Euclidean distance
        o_dist = np.sqrt(np.sum((o_u - o_v) ** 2))
        d_dist = np.sqrt(np.sum((d_u - d_v) ** 2))

        # Combine using the specified distance metric
        if distance == 'max':
            return max(o_dist, d_dist)
        elif distance == 'min':
            return min(o_dist, d_dist)
        elif distance == 'sum':
            return o_dist + d_dist
        elif distance == 'mean':
            return (o_dist + d_dist) / 2

    return flow_metric


class DBSCANFlow:
    """DBSCAN clustering for flow data using flow-specific distance metrics.

    This class implements DBSCAN clustering adapted for flow data, where
    the distance metric between flows is based on their spatial characteristics
    (origin and destination points).

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples to be considered neighbors.
    min_samples : int, default=5
        Number of samples in a neighborhood for a point to be considered
        a core point.
    distance : str, default='max'
        Distance metric for flows: 'max', 'min', 'sum', 'mean'.
        - 'max': Maximum of origin and destination distances
        - 'min': Minimum of origin and destination distances
        - 'sum': Sum of origin and destination distances
        - 'mean': Average of origin and destination distances
    **kwargs
        Additional keyword arguments passed to sklearn.cluster.DBSCAN
        (e.g., n_jobs, leaf_size).

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each flow. -1 indicates noise.
    n_clusters_ : int
        Number of clusters found.
    core_sample_indices_ : np.ndarray
        Indices of core samples.

    Examples
    --------
    >>> from geoflowkit import read_file
    >>> from geoflowkit.clustering import DBSCANFlow
    >>> fdf = read_file('flows.gpkg')
    >>> db = DBSCANFlow(eps=0.5, min_samples=5, distance='max')
    >>> db.fit(fdf)
    >>> print(db.labels_)
    """

    DISTANCE_OPTIONS = {'max', 'min', 'sum', 'mean'}

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 distance: str = 'max', **kwargs):
        if distance not in self.DISTANCE_OPTIONS:
            raise ValueError(
                f"distance must be one of {self.DISTANCE_OPTIONS}, "
                f"got {distance}"
            )
        self.eps = eps
        self.min_samples = min_samples
        self.distance = distance
        self.kwargs = kwargs
        self.labels_ = None
        self.n_clusters_ = None
        self.core_sample_indices_ = None

    def fit(self, fdf: FlowDataFrame) -> 'DBSCANFlow':
        """Perform DBSCAN clustering on flow data.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        self
            Fitted estimator.
        """
        # Build feature matrix: [o_x, o_y, d_x, d_y]
        X = np.column_stack([
            fdf.o.x.values,
            fdf.o.y.values,
            fdf.d.x.values,
            fdf.d.y.values
        ])

        # Create flow distance function
        metric_func = _flow_distance_factory(distance=self.distance)

        # Run sklearn DBSCAN
        db = SklearnDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=metric_func,
            **self.kwargs
        )
        db.fit(X)

        self.labels_ = db.labels_
        self.core_sample_indices_ = db.core_sample_indices_
        # Calculate number of clusters (excluding noise label -1)
        self.n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        return self

    def fit_predict(self, fdf: FlowDataFrame) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(fdf)
        return self.labels_

    @property
    def n_clusters(self) -> int:
        """Number of clusters found (excluding noise)."""
        if self.n_clusters_ is None:
            raise ValueError("Model must be fitted before accessing n_clusters_")
        return self.n_clusters_


def dbscan(fdf: FlowDataFrame, eps: float = 0.5, min_samples: int = 5,
           distance: str = 'max', **kwargs) -> np.ndarray:
    """Perform DBSCAN clustering on flow data using flow-specific distance metrics.

    This is a convenience function that creates a DBSCANFlow instance,
    fits it, and returns the cluster labels.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe.
    eps : float, optional
        Maximum distance between two samples to be considered neighbors,
        by default 0.5.
    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered
        a core point, by default 5.
    distance : str, optional
        Distance metric: 'max', 'min', 'sum', 'mean', by default 'max'.
    **kwargs
        Additional keyword arguments passed to DBSCANFlow.

    Returns
    -------
    np.ndarray
        Cluster labels for each flow. -1 indicates noise.

    Examples
    --------
    >>> from geoflowkit import read_file
    >>> from geoflowkit.clustering import dbscan
    >>> fdf = read_file('flows.gpkg')
    >>> labels = dbscan(fdf, eps=0.5, min_samples=5, distance='max')
    """
    model = DBSCANFlow(
        eps=eps,
        min_samples=min_samples,
        distance=distance,
        **kwargs
    )
    return model.fit_predict(fdf)
