"""K-medoid clustering for FlowDataFrame objects using flow distance metrics."""

import numpy as np
import shapely

from geoflowkit import FlowDataFrame
from geoflowkit.flowmetrics import pairwise_distances


class KMedoidFlow:
    """K-medoid clustering for flow data using flow-specific distance metrics.

    This class implements the K-medoid algorithm adapted for flows, where
    the distance metric between flows is based on their spatial characteristics
    (origin and destination points).

    Supports two computation modes:
    - 'precompute': Precompute full distance matrix (faster for small datasets)
    - 'online': Compute distances on-the-fly (memory efficient for large datasets)

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters.
    distance : str, default='max'
        The distance metric to use. Options are:
        - 'max': Maximum of origin and destination distances
        - 'min': Minimum of origin and destination distances
        - 'sum': Sum of origin and destination distances
        - 'mean': Average of origin and destination distances
    method : str, default='auto'
        Computation method. Options are:
        - 'auto': Automatically choose based on data size
        - 'precompute': Always precompute full distance matrix
        - 'online': Always compute distances on-the-fly
    medoid_sample_size : int, default=100
        Number of candidates to sample when searching for medoid (CLARANS style).
    max_iter : int, default=100
        Maximum number of iterations.
    n_init : int, default=10
        Number of times to run the algorithm with different medoid seeds.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each flow.
    cluster_centers_ : np.ndarray
        Indices of medoid samples in the input data.
    inertia_ : float
        Sum of distances from each sample to its cluster medoid.
    """

    # Threshold for auto mode: if n_samples <= AUTO_THRESHOLD, use precompute
    AUTO_THRESHOLD = 5000

    def __init__(self, n_clusters: int = 5, distance: str = 'max',
                 method: str = 'auto',
                 medoid_sample_size: int = 100,
                 max_iter: int = 100, n_init: int = 10, random_state: int = None):
        self.n_clusters = n_clusters
        self.distance = distance
        self.method = method
        self.medoid_sample_size = medoid_sample_size
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self._fdf = None
        self._dis_matrix = None
        self._use_precompute = None

    def fit(self, fdf: FlowDataFrame) -> 'KMedoidFlow':
        """Fit the K-medoid model to flow data.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        self
            Fitted estimator.
        """
        self._fdf = fdf
        n_samples = fdf.shape[0]

        # Determine computation method
        if self.method == 'auto':
            self._use_precompute = n_samples <= self.AUTO_THRESHOLD
        elif self.method == 'precompute':
            self._use_precompute = True
        elif self.method == 'online':
            self._use_precompute = False
        else:
            raise ValueError(f"Unknown method: {self.method}. "
                           f"Must be 'auto', 'precompute', or 'online'.")

        # Precompute distance matrix if using precompute mode
        if self._use_precompute:
            self._dis_matrix = pairwise_distances(fdf, distance=self.distance)
        else:
            self._dis_matrix = None

        best_labels = None
        best_centers = None
        best_inertia = np.inf

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_init):
            # Initialize cluster medoids
            centers = self._init_centers(n_samples, rng)

            for _ in range(self.max_iter):
                # Assign each flow to nearest medoid
                labels = self._assign_to_clusters(centers, n_samples)

                # Update cluster medoids (CLARANS-style sampling)
                new_centers = self._update_centers(labels, n_samples, rng)

                # Check convergence
                if np.array_equal(centers, new_centers):
                    centers = new_centers
                    break
                centers = new_centers

            # Calculate inertia
            inertia = self._calculate_inertia(labels, centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()

        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia

        return self

    def _get_dist(self, i: int, j: int) -> float:
        """Get distance between flow i and j."""
        if self._use_precompute:
            return self._dis_matrix[i, j]
        else:
            o_i = self._fdf.o.iloc[i]
            d_i = self._fdf.d.iloc[i]
            o_j = self._fdf.o.iloc[j]
            d_j = self._fdf.d.iloc[j]
            return self._compute_dist(o_i, d_i, o_j, d_j)

    def _compute_dist(self, o_i, d_i, o_j, d_j) -> float:
        """Compute distance between two flows."""
        o_dist = shapely.distance(o_i, o_j)
        d_dist = shapely.distance(d_i, d_j)

        if self.distance == 'max':
            return max(o_dist, d_dist)
        elif self.distance == 'sum':
            return o_dist + d_dist
        elif self.distance == 'min':
            return min(o_dist, d_dist)
        elif self.distance == 'mean':
            return (o_dist + d_dist) / 2
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def _init_centers(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        """Initialize medoid indices using k-means++ style initialization."""
        centers = np.zeros(self.n_clusters, dtype=int)
        centers[0] = rng.randint(0, n_samples)

        for i in range(1, self.n_clusters):
            dist_to_nearest = np.array([
                min(self._get_dist(j, centers[c]) for c in range(i))
                for j in range(n_samples)
            ])

            probs = dist_to_nearest ** 2
            probs /= probs.sum()
            centers[i] = rng.choice(n_samples, p=probs)

        return centers

    def _assign_to_clusters(self, centers: np.ndarray, n_samples: int) -> np.ndarray:
        """Assign each sample to the nearest medoid."""
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            min_dist = np.inf
            for j, c in enumerate(centers):
                d = self._get_dist(i, c)
                if d < min_dist:
                    min_dist = d
                    labels[i] = j

        return labels

    def _update_centers(self, labels: np.ndarray, n_samples: int,
                       rng: np.random.RandomState) -> np.ndarray:
        """Update cluster medoids using CLARANS-style sampling."""
        centers = np.zeros(self.n_clusters, dtype=int)

        for k in range(self.n_clusters):
            cluster_indices = np.where(labels == k)[0]
            cluster_size = len(cluster_indices)

            if cluster_size == 0:
                centers[k] = rng.randint(0, n_samples)
                continue

            # CLARANS-style: sample candidates
            sample_size = min(self.medoid_sample_size, cluster_size)
            candidates = rng.choice(cluster_indices, size=sample_size, replace=False)

            other_mask = np.ones(n_samples, dtype=bool)
            other_mask[cluster_indices] = False
            other_indices = np.where(other_mask)[0]

            if len(other_indices) == 0:
                centers[k] = cluster_indices[0]
                continue

            best_candidate = candidates[0]
            best_sum_dist = np.inf

            for candidate in candidates:
                if self._use_precompute:
                    sum_dist = self._dis_matrix[candidate, other_indices].sum()
                else:
                    sum_dist = sum(self._get_dist(candidate, o) for o in other_indices)

                if sum_dist < best_sum_dist:
                    best_sum_dist = sum_dist
                    best_candidate = candidate

            centers[k] = best_candidate

        return centers

    def _calculate_inertia(self, labels: np.ndarray, centers: np.ndarray) -> float:
        """Calculate sum of distances from each sample to its cluster medoid."""
        return sum(
            self._get_dist(i, centers[labels[i]])
            for i in range(len(labels))
        )

    def fit_predict(self, fdf: FlowDataFrame) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(fdf)
        return self.labels_

    def predict(self, _: FlowDataFrame) -> np.ndarray:
        """Predict cluster labels (returns fitted labels)."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.labels_


def kmedoid(fdf: FlowDataFrame, n_clusters: int = 5, distance: str = 'max',
            method: str = 'auto', medoid_sample_size: int = 100,
            max_iter: int = 100, n_init: int = 10,
            random_state: int = None) -> np.ndarray:
    """Perform K-medoid clustering on flow data using flow-specific distance metrics.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe.
    n_clusters : int, optional
        Number of clusters, by default 5.
    distance : str, optional
        The distance metric to use ('max', 'min', 'sum', 'mean'), by default 'max'.
    method : str, optional
        Computation method: 'auto', 'precompute', or 'online'.
        - 'auto': Automatically choose based on data size (default)
        - 'precompute': Precompute full distance matrix (faster for small datasets)
        - 'online': Compute distances on-the-fly (memory efficient for large datasets)
    medoid_sample_size : int, optional
        Number of candidates to sample when searching for medoid (CLARANS style),
        by default 100.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    n_init : int, optional
        Number of initializations, by default 10.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Cluster labels for each flow.

    Examples
    --------
    >>> from geoflowkit import FlowDataFrame, kmedoid
    >>> fdf = FlowDataFrame.from_csv('flows.csv', ...)
    >>> labels = kmedoid(fdf, n_clusters=3, distance='max')
    """
    model = KMedoidFlow(
        n_clusters=n_clusters,
        distance=distance,
        method=method,
        medoid_sample_size=medoid_sample_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state
    )
    return model.fit_predict(fdf)
