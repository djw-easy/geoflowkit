"""DBSCAN clustering for FlowDataFrame objects."""

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from geoflowkit import FlowDataFrame


def dbscan(fdf: FlowDataFrame, eps: float = 0.5, min_samples: int = 5, **kwargs) -> np.ndarray:
    """Perform DBSCAN clustering on flow data.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe
    eps : float, optional
        Maximum distance between two samples to be considered neighbors, by default 0.5
    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered a core point, by default 5
    **kwargs
        Additional keyword arguments for sklearn.cluster.DBSCAN

    Returns
    -------
    np.ndarray
        Cluster labels for each flow (-1 indicates noise)

    Examples
    --------
    >>> from geoflowkit import FlowDataFrame, dbscan
    >>> fdf = FlowDataFrame.from_csv('flows.csv', ...)
    >>> labels = dbscan(fdf, eps=0.5, min_samples=5)
    """
    # Extract flow features (origin, destination coordinates)
    o_coords = np.column_stack([
        fdf.o.x.values,
        fdf.o.y.values
    ])
    d_coords = np.column_stack([
        fdf.d.x.values,
        fdf.d.y.values
    ])

    # Combine origin and destination features
    X = np.concatenate([o_coords, d_coords], axis=1)

    # Run DBSCAN
    dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    return dbscan.fit_predict(X)
