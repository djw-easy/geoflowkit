"""K-Means clustering for FlowDataFrame objects using flow-specific distance metrics."""

import numpy as np

from geoflowkit import FlowDataFrame, FlowSeries, Flow


class KMeansFlow:
    """K-Means clustering for flow data using flow-specific distance metrics.

    Cluster centers are represented as virtual flows (4D means of assigned flows),
    where the origin and destination points are the mean coordinates of all flows
    in each cluster. Assignment uses flow-specific distance metrics (max/min/sum/mean),
    while center update computes the mean of assigned 4D flow vectors.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    distance : str, default='max'
        The distance metric for flow-to-center assignment. Options are:
        - 'max': Maximum of origin and destination distances
        - 'min': Minimum of origin and destination distances
        - 'sum': Sum of origin and destination distances
        - 'mean': Average of origin and destination distances
    init : str or array-like, default='k-means++'
        Initialization method:
        - 'k-means++': Select initial centers using k-means++ heuristic
        - 'random': Select random flows as initial centers
        - array-like of shape (n_clusters, 4): User-specified initial centers
          as [o_x, o_y, d_x, d_y] vectors
    n_init : int, default=10
        Number of initializations to run. The best result (lowest inertia) is kept.
    max_iter : int, default=300
        Maximum number of iterations per initialization.
    tol : float, default=1e-4
        Relative tolerance for convergence (mean center displacement).
    random_state : int, optional
        Random seed for reproducible results.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each flow.
    cluster_centers_ : FlowSeries
        Cluster centers as Flow objects (virtual flows).
    inertia_ : float
        Sum of distances from each sample to its cluster center.
    n_iter_ : int
        Number of iterations run in the best initialization.

    Examples
    --------
    >>> from geoflowkit import read_file
    >>> from geoflowkit.clustering import KMeansFlow
    >>> fdf = read_file('flows.gpkg')
    >>> km = KMeansFlow(n_clusters=5, distance='max', random_state=42)
    >>> km.fit(fdf)
    >>> print(km.labels_)
    """

    DISTANCE_OPTIONS = {'max', 'min', 'sum', 'mean'}

    def __init__(self, n_clusters=8, distance='max', init='k-means++',
                 n_init=10, max_iter=300, tol=1e-4, random_state=None):
        if distance not in self.DISTANCE_OPTIONS:
            raise ValueError(
                f"distance must be one of {self.DISTANCE_OPTIONS}, "
                f"got {distance}"
            )
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if isinstance(init, str):
            if init not in ('k-means++', 'random'):
                raise ValueError(
                    f"init must be 'k-means++', 'random', or an array-like, "
                    f"got {init}"
                )
        else:
            init = np.asarray(init, dtype=float)
            if init.ndim != 2 or init.shape != (n_clusters, 4):
                raise ValueError(
                    f"init array must have shape ({n_clusters}, 4), "
                    f"got {init.shape}"
                )
        self.n_clusters = n_clusters
        self.distance = distance
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self._centers = None

    def fit(self, fdf: FlowDataFrame) -> 'KMeansFlow':
        """Perform K-Means clustering on flow data.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = np.column_stack([
            fdf.o.x.values,
            fdf.o.y.values,
            fdf.d.x.values,
            fdf.d.y.values
        ])
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} is larger than the number "
                f"of samples ({n_samples})"
            )

        best_inertia = np.inf
        best_labels = None
        best_centers = None
        best_n_iter = 0

        rng = np.random.RandomState(self.random_state)

        for run in range(self.n_init):
            centers = self._init_centers(X, rng)
            prev_centers = np.empty_like(centers)

            for n_iter in range(1, self.max_iter + 1):
                prev_centers[:] = centers

                labels = self._assign(X, centers)
                centers = self._update_centers(X, labels, centers)

                shift = np.mean(
                    np.sqrt(np.sum((centers - prev_centers) ** 2, axis=1))
                )
                if shift < self.tol:
                    break

            inertia = self._compute_inertia(X, labels, centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_n_iter = n_iter

        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self._centers = best_centers
        self.cluster_centers_ = self._centers_to_flowseries(best_centers)

        return self

    def fit_predict(self, fdf: FlowDataFrame) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(fdf)
        return self.labels_

    def transform(self, fdf: FlowDataFrame) -> np.ndarray:
        """Compute distances from each flow to each cluster center.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        np.ndarray of shape (n_samples, n_clusters)
            Distance matrix. Row i is the distances from flow i to each center.
        """
        if self._centers is None:
            raise ValueError("Model must be fitted before calling transform.")
        X = np.column_stack([fdf.o.x.values, fdf.o.y.values,
                             fdf.d.x.values, fdf.d.y.values])
        return self._pairwise_distances(X, self._centers)

    def _init_centers(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Initialize cluster centers."""
        if isinstance(self.init, str):
            if self.init == 'k-means++':
                return self._kmeans_plus_plus(X, rng)
            elif self.init == 'random':
                indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
                return X[indices].copy()
            else:
                raise ValueError(f"Unknown init method: {self.init}")
        else:
            init = np.asarray(self.init, dtype=float)
            if init.ndim != 2 or init.shape != (self.n_clusters, 4):
                raise ValueError(
                    f"init array must have shape ({self.n_clusters}, 4), "
                    f"got {init.shape}"
                )
            return init.copy()

    def _kmeans_plus_plus(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Initialize centers using k-means++ heuristic."""
        centers = np.empty((self.n_clusters, 4))
        centers[0] = X[rng.randint(X.shape[0])]

        for i in range(1, self.n_clusters):
            dists = self._pairwise_distances(X, centers[:i])
            min_dists = dists.min(axis=1)
            probs = min_dists ** 2
            probs /= probs.sum()
            centers[i] = X[rng.choice(X.shape[0], p=probs)]

        return centers

    def _pairwise_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute distances from all samples to all centers.

        Returns array of shape (n_samples, n_centers).
        """
        o_dists = np.sqrt(
            np.sum((X[:, None, :2] - centers[None, :, :2]) ** 2, axis=2)
        )
        d_dists = np.sqrt(
            np.sum((X[:, None, 2:] - centers[None, :, 2:]) ** 2, axis=2)
        )

        if self.distance == 'max':
            return np.maximum(o_dists, d_dists)
        elif self.distance == 'min':
            return np.minimum(o_dists, d_dists)
        elif self.distance == 'sum':
            return o_dists + d_dists
        elif self.distance == 'mean':
            return (o_dists + d_dists) / 2

    def _assign(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest center."""
        dists = self._pairwise_distances(X, centers)
        return np.argmin(dists, axis=1).astype(np.int32)

    def _update_centers(self, X: np.ndarray, labels: np.ndarray,
                        prev_centers: np.ndarray) -> np.ndarray:
        """Update centers as the mean of assigned samples."""
        centers = np.empty((self.n_clusters, 4))
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = prev_centers[k]
        return centers

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                         centers: np.ndarray) -> float:
        """Compute sum of distances from each sample to its assigned center."""
        dists = self._pairwise_distances(X, centers)
        return float(dists[np.arange(len(labels)), labels].sum())

    def _centers_to_flowseries(self, centers: np.ndarray) -> FlowSeries:
        """Convert 4D center vectors to a FlowSeries."""
        flows = [Flow([[c[0], c[1]], [c[2], c[3]]]) for c in centers]
        return FlowSeries(flows)


def kmeans(fdf: FlowDataFrame, n_clusters=8, distance='max', init='k-means++',
           n_init=10, max_iter=300, tol=1e-4,
           random_state=None) -> np.ndarray:
    """Perform K-Means clustering on flow data using flow-specific distance metrics.

    This is a convenience function that creates a KMeansFlow instance,
    fits it, and returns the cluster labels.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe.
    n_clusters : int, optional
        Number of clusters, by default 8.
    distance : str, optional
        The distance metric ('max', 'min', 'sum', 'mean'), by default 'max'.
    init : str or array-like, optional
        Initialization method, by default 'k-means++'.
    n_init : int, optional
        Number of initializations, by default 10.
    max_iter : int, optional
        Maximum iterations per initialization, by default 300.
    tol : float, optional
        Convergence tolerance, by default 1e-4.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Cluster labels for each flow.

    Examples
    --------
    >>> from geoflowkit import FlowDataFrame
    >>> from geoflowkit.clustering import kmeans
    >>> fdf = FlowDataFrame.from_csv('flows.csv', ...)
    >>> labels = kmeans(fdf, n_clusters=3, distance='max', random_state=42)
    """
    model = KMeansFlow(
        n_clusters=n_clusters,
        distance=distance,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    return model.fit_predict(fdf)
