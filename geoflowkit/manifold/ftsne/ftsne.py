import time
import warnings
import numpy as np
import pandas as pd
from numbers import Integral, Real
from typing import Optional, Union, Tuple


from shapely import get_coordinates
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.manifold._t_sne import _joint_probabilities, _joint_probabilities_nn
from sklearn.utils._param_validation import Hidden, Interval, StrOptions, validate_params


from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.manifold.ftsne.utils import (
    calc_optimized_p_cond, 
    calc_optimized_p_cond_nn, 
    get_multivariate_p_cond, 
    get_multivariate_p_cond_nn, 
    kl_grad, kl_grad_bh, 
    hd_grad,
    GDOptimizer
)


class FTSNE:
    """ft-SNE: A Variant of t-SNE for Visualizing Geographical Flow Data

    ft-SNE (Flow t-SNE) is a dimensionality reduction method specifically designed
    for visualizing geographical flow data. It extends the classical t-SNE algorithm
    by modeling flow characteristics such as origin-destination pairs, flow length,
    and directional information in a unified embedding space.

    Unlike standard t-SNE which treats data points independently, ft-SNE incorporates
    the relational structure of flows through three types of mappings:

    - **Identity mapping**: Maps a single flow attribute (e.g., origin coordinates)
      directly to an embedding dimension
    - **Intersection mapping**: Combines multiple attributes (e.g., both origin and
      destination coordinates) to model joint flow probability
    - **Union mapping**: Aggregates multiple attributes to capture overall flow similarity

    The algorithm optimizes aKL divergence or Hellinger distance loss between the
    probability distributions in high-dimensional flow space and the low-dimensional
    embedding space.

    Parameters
    ----------
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.
    learning_rate : float or 'auto', default='auto'
        The initial learning rate for the embedding optimization.
        The 'auto' option sets the learning_rate
        to `max(N / early_exaggeration / 4, 50)` where N is the sample size, following [4] and [5].
    max_iter : int, default=100
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    early_exaggeration_iter : int or 'auto', default='auto'
        Number of training cycles in which exaggeration will be applied.
        The 'auto' option sets this to `max_iter // 4`.
    init : str or np.ndarray, default='pca'
        Method to use for initialization of the embedding. Options are:

        - 'pca': Initialize using PCA on flow attributes (recommended)
        - 'random': Initialize with random values
        - np.ndarray: Use a custom initial embedding matrix of shape (n_samples, n_components)
    method : {'exact', 'barnes_hut'}, default='exact'
        The algorithm to use for computing the gradient:

        - 'exact': Use the exact gradient computation (O(n^2), suitable for n < 5000)
        - 'barnes_hut': Use the Barnes-Hut approximation (O(n log n), for larger datasets)

        Note: 'barnes_hut' does not yet support intersection and union mappings.
    random_state : int, default=None
        Seed for random number generator. If None, the random state is not set.
    loss_func : {'kl', 'hd'}, default='kl'
        Loss function to use for optimization:

        - 'kl': Kullback-Leibler divergence (default)
        - 'hd': Hellinger distance (requires method='exact')
    metric : str, default='euclidean'
        The metric to use to compute distances in high dimensional space.
        Valid string metrics include:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
          'nan_euclidean']
        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation',
          'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
          'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these metrics.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    log_progress : {'tqdm', 'notebook', 'none'}, default='tqdm'
        Method to use for logging progress:

        - 'tqdm': Use tqdm progress bar (for console)
        - 'notebook': Use tqdm.notebook (for Jupyter notebooks)
        - 'none': Disable progress logging
    n_jobs : int, default=None
        The number of parallel jobs to run for pairwise distances calculation.
        None means 1 job unless explicitly set otherwise.
    verbose : int, default=0
        Verbosity level. If non-zero, progress is printed to stdout.
        Higher values produce more detailed output.
    angle : float, default=0.5
        Only used if method='barnes_hut'. This is the trade-off between speed and
        accuracy for Barnes-Hut T-SNE. 'angle' is the angular size (referred to as
        theta in [3]) of a distant node as measured from a point. If this size is
        below 'angle' then it is used as a summary node of all points contained
        within it. This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    Attributes
    ----------
    n_components : int
        The number of embedding dimensions (set during fit).
    n_samples : int
        The number of samples in the input data (set during fit).
    error_ : float
        The final optimization error (KL divergence or Hellinger distance).
    perplexity_ : float
        The effective perplexity used (may differ from input if adjusted).

    Examples
    --------
    >>> from geoflowkit import FlowDataFrame
    >>> from geoflowkit.manifold import FTSNE

    Basic usage with origin-destination flow data:

    >>> fdf = FlowDataFrame.from_path("flows.csv", origin="origin", destination="dest")
    >>> ft = FTSNE(perplexity=30, learning_rate='auto', random_state=42)
    >>> embedding = ft.fit_transform(
    ...     fdf,
    ...     identity={'o': 0, 'd': 1}  # Map origin and destination to dims 0 and 1
    ... )

    Using intersection mapping for joint origin-destination modeling:

    >>> embedding = ft.fit_transform(
    ...     fdf,
    ...     intersection={('o', 'd'): 0}  # Joint OD modeling on dimension 0
    ... )

    Using union mapping to aggregate multiple flow attributes:

    >>> embedding = ft.fit_transform(
    ...     fdf,
    ...     union={('o', 'd'): (0, 1)}  # OD union on dimensions 0 and 1
    ... )

    Notes
    -----
    The ft-SNE algorithm consists of the following steps:

    1. **Parameter validation**: Check that mappings and metrics are valid
    2. **Embedding initialization**: Initialize low-dimensional embedding using PCA,
       random initialization, or provided matrix
    3. **Distance computation**: Compute pairwise distances for each flow attribute
    4. **Probability computation**: Convert distances to probability distributions using
       perplexity-based Gaussian kernels
    5. **Optimization**: Iterate to minimize the divergence between high-dimensional
       and embedding probability distributions

    References
    ----------
    [1] Maaten, L. van der, & Hinton, G. (2008). Visualizing Data using t-SNE.
        Journal of Machine Learning Research, 9, 2579-2605.
    [2] Van der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms.
        Journal of Machine Learning Research, 15(1), 3221-3245.
    """
    _parameter_constraints: dict = {
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "max_iter": [Interval(Integral, 10, None, closed="left"), None],
        "early_exaggeration": [Interval(Real, 1, None, closed="left")],
        "early_exaggeration_iter": [Interval(Integral, 0, None, closed="left"), None],
        "init": [
            StrOptions({"pca", "random"}),
            np.ndarray,
        ],
        "method": [StrOptions({"barnes_hut", "exact"})],
        "random_state": ["random_state"],
        "loss_func": [StrOptions({"kl", "hd"})],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "log_progress": [StrOptions({"tqdm", "notebook", "none"})],
        "n_jobs": [None, Integral],
        "verbose": [Interval(Integral, 0, None, closed="left")],
        "angle": [Interval(Real, 0, 1, closed="both")],
    }

    @validate_params(_parameter_constraints, prefer_skip_nested_validation=True)
    def __init__(self, 
                 perplexity=30.0, 
                 learning_rate=0.1,
                 max_iter=100, 
                 early_exaggeration=12.0,
                 early_exaggeration_iter='auto', 
                 init='pca', 
                 method="exact",
                 random_state=None, 
                 loss_func='kl', 
                 metric='euclidean', 
                 metric_params=None,
                 log_progress='tqdm', 
                 n_jobs=None, 
                 verbose=0, 
                 angle=0.5):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        if early_exaggeration_iter == 'auto':
            self.early_exaggeration_iter = self.max_iter // 4
        self.init = init
        self.method = method
        self.random_state = random_state
        self.loss_func = loss_func
        self.metric = metric
        self.metric_params = metric_params
        self.log_progress = log_progress
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.angle = angle

    def _init_pbar(self, iter_num):
        if self.log_progress == 'tqdm':
            from tqdm import tqdm
            self.pbar = tqdm(total=iter_num)
        elif self.log_progress == 'notebook':
            from tqdm.notebook import tqdm
            self.pbar = tqdm(total=iter_num)
        else:
            self.pbar = None

    def _check_params(self, fdf: Union[FlowDataFrame, dict], identity: dict = None,
                      intersection: dict = None, union: dict = None,
                      metrics: dict = None) -> int:
        """Check parameters and calculate embedding dimension.

        This method validates all input parameters including the flow data,
        mapping dictionaries, and metric specifications. It ensures that:

        1. The FlowDataFrame or dict contains valid data
        2. All mapping attributes reference valid columns/attributes
        3. Dimensions are continuous (no gaps) and properly specified
        4. Metrics are valid and reference used attributes

        Parameters
        ----------
        fdf : Union[FlowDataFrame, dict]
            Input flow data. If dict, must contain 2D numpy arrays as values
            with the same number of rows (samples) and columns (features).
        identity : dict, optional
            Identity mapping that maps single features to embedding dimensions.
            Keys are attribute names, values are dimension indices (int) or
            tuples of indices for polynomial features.
            Example: {'o': 0, 'd': 1}
        intersection : dict, optional
            Intersection mapping that combines multiple features to model
            joint flow probability. Keys are tuples of attribute names,
            values are dimension indices.
            Example: {('o', 'd'): 0}
        union : dict, optional
            Union mapping that aggregates multiple features. Keys are tuples
            of attribute names, values are dimension indices.
            Example: {('o', 'd'): (0, 1)}
        metrics : dict, optional
            Metric specification for each attribute. Keys are attribute names,
            values are valid metric strings.
            Example: {'o': 'euclidean', 'd': 'haversine'}

        Returns
        -------
        int
            The calculated embedding dimension (max_dim + 1).

        Raises
        ------
        ValueError
            If any parameter is invalid, including:
            - Precomputed values are not 2D numpy arrays
            - Precomputed arrays have mismatched dimensions
            - Mapping attributes are not valid
            - Dimensions are not continuous
            - Metrics reference unused attributes or are invalid

        Examples
        --------
        >>> ft = FTSNE()
        >>> n_components = ft._check_params(
        ...     fdf,
        ...     identity={'o': 0, 'd': 1},
        ...     intersection={('o', 'd'): 2}
        ... )
        >>> print(n_components)
        3
        """
        # Validate fdf type and get valid attributes
        if isinstance(fdf, dict):
            # Validate dict values
            n_samples = None
            for key, value in fdf.items():
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"Precomputed value for key {key} must be numpy array")
                if value.ndim != 2:
                    raise ValueError(f"Precomputed value for key {key} must be 2D array")
                if n_samples is None:
                    n_samples = value.shape[0]
                if value.shape[0] != n_samples or value.shape[1] != 2:
                    raise ValueError(f"All Precomputed values must have same number of rows and columns, got {value.shape[0]} for key {key} (expected {n_samples})")
            
            valid_attrs = set(fdf.keys())
        else:
            # FlowDataFrame case
            valid_attrs = set(fdf.columns) | {'o', 'd', 'length', 'angle'}
        
        # Collect all dimensions used in mappings
        dims = set()
        
        # Check identity mapping
        if identity is not None:
            for attr, dims_tuple in identity.items():
                if attr not in valid_attrs:
                    raise ValueError(f"Invalid identity attribute: {attr}")
                if isinstance(dims_tuple, (tuple, list)):
                    assert len(set(dims_tuple)) == len(dims_tuple), f"Duplicate identity dimension: {dims_tuple}"
                    for dim in dims_tuple:
                        if dim in dims:
                            raise ValueError(f"Duplicate identity dimension: {dim}")
                    dims.update(dims_tuple)
                if isinstance(dims_tuple, int):
                    dims.add(dims_tuple)
                
        # Check intersection mapping
        if intersection is not None:
            for attrs, dims_tuple in intersection.items():
                if not isinstance(attrs, tuple):
                    raise ValueError(f"Intersection key must be tuple, got {type(attrs)}")
                for attr in attrs:
                    if attr not in valid_attrs:
                        raise ValueError(f"Invalid intersection attribute: {attr}")
                if isinstance(dims_tuple, int):
                    dims.add(dims_tuple)
                elif isinstance(dims_tuple, (tuple, list)):
                    assert len(set(dims_tuple)) == len(dims_tuple), f"Duplicate identity dimension: {dims_tuple}"
                    dims.update(dims_tuple)
                else:
                    raise ValueError(f"Intersection value must be int or tuple, got {type(dims_tuple)}")
                    
        # Check union mapping
        if union is not None:
            for attrs, dims_tuple in union.items():
                if not isinstance(attrs, tuple):
                    raise ValueError(f"Union key must be tuple, got {type(attrs)}")
                for attr in attrs:
                    if attr not in valid_attrs:
                        raise ValueError(f"Invalid union attribute: {attr}")
                if isinstance(dims_tuple, int):
                    dims.add(dims_tuple)
                elif isinstance(dims_tuple, (tuple, list)):
                    assert len(set(dims_tuple)) == len(dims_tuple), f"Duplicate identity dimension: {dims_tuple}"
                    dims.update(dims_tuple)
                else:
                    raise ValueError(f"Union value must be int or tuple, got {type(dims_tuple)}")
                    
        # Calculate and validate embedding dimension
        if not dims:
            raise ValueError("At least one dimension must be specified in identity, intersection or union mappings")
        
        min_dim = min(dims)
        max_dim = max(dims)
        expected_dims = set(range(min_dim, max_dim + 1))
        
        if dims != expected_dims:
            raise ValueError(f"Dimensions must be continuous, got {dims}")
            
        self.n_components = max_dim + 1

        # Check metrics mapping
        if metrics is not None:
            # Get all attributes used in identity/intersection/union
            used_attrs = set()
            if identity is not None:
                used_attrs.update(identity.keys())
            if intersection is not None:
                for attrs in intersection.keys():
                    used_attrs.update(attrs)
            if union is not None:
                for attrs in union.keys():
                    used_attrs.update(attrs)
            
            for attr, metric in metrics.items():
                if attr not in used_attrs:
                    raise ValueError(f"Metrics attribute {attr} must be used in identity, intersection or union")
                if metric not in _VALID_METRICS:
                    raise ValueError(f"Invalid metric {metric}. Valid metrics are: {_VALID_METRICS}")

        return self.n_components
    
    def _get_values(self, fdf: FlowDataFrame, attr: Union[str, tuple]):
        """Extract feature values from flow data by attribute name.

        This method retrieves the numerical values for a given attribute from the
        FlowDataFrame. It handles special cases for flow-specific attributes ('o',
        'd', 'length', 'angle') and regular column attributes.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow data frame containing flow information.
        attr : str or tuple of str
            The attribute name(s) to extract values for. Can be:

            - Regular column name (e.g., 'population', 'category')
            - Flow attribute: 'o' (origin), 'd' (destination),
              'length' (flow length), 'angle' (flow direction)

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (n_samples, n_features) containing
            the extracted values. For single attributes, this is (n_samples, 1).
            For multiple attributes passed as a tuple, values are concatenated
            along the feature axis.

        Raises
        ------
        ValueError
            If the attribute is not found in the FlowDataFrame columns or
            is not a valid flow attribute ('o', 'd', 'length', 'angle').

        Examples
        --------
        >>> values = ft._get_values(fdf, 'o')  # Get origin coordinates
        >>> values = ft._get_values(fdf, 'length')  # Get flow lengths
        >>> values = ft._get_values(fdf, ('o', 'd'))  # Get both origin and destination
        """
        
    def _initialize_embedding(self, fdf: Union[FlowDataFrame, dict], identity: dict,
                            intersection: dict, union: dict, n_components: int):
        """Initialize the embedding matrix.

        Creates an initial low-dimensional embedding for the optimization process.
        The initialization method can be PCA-based (recommended), random, or
        a provided custom matrix.

        When using PCA initialization with identity/intersection/union mappings,
        each attribute is individually scaled to [0, 1] range and reduced to
        the specified embedding dimensions using PCA or polynomial features.

        Parameters
        ----------
        fdf : Union[FlowDataFrame, dict]
            Input flow data. If dict, must contain 2D numpy arrays as values
            with the same number of rows (samples).
        identity : dict
            Identity mapping that maps single features to embedding dimensions.
        intersection : dict
            Intersection mapping that combines multiple features.
        union : dict
            Union mapping that aggregates multiple features.
        n_components : int
            Number of embedding dimensions (determined from mappings).

        Returns
        -------
        np.ndarray
            Initial embedding matrix of shape (n_samples, n_components).

        Notes
        -----
        The initialization priority is: identity > intersection > union.
        If a dimension is already assigned by identity mapping, intersection
        and union mappings will skip that dimension and use random initialization
        instead.

        When using polynomial features (tuple of dimensions as value), the
        polynomial degree equals the number of dimensions, creating interaction
        terms between the original features.
        """
        # Get number of samples
        n_samples = len(fdf) if isinstance(fdf, FlowDataFrame) else next(iter(fdf.values())).shape[0]
        self.n_samples = n_samples

        # Initialize random state
        np.random.seed(self.random_state)
        
        # Initialize embedding matrix
        embedding = 1e-4 * np.random.randn(n_samples, n_components)
        
        if isinstance(fdf, dict) and self.init == 'pca':
            print("PCA initialization is not supported for precomputed input, using random initialization instead")
            return embedding
        elif self.init == 'pca':
            scaler = MinMaxScaler()
            # Check for dimension conflicts
            identity_used_dims = set()
            
            # Process identity mapping
            if identity is not None:
                for attr, dim in identity.items():
                    values = self._get_values(fdf, attr)
                    values = scaler.fit_transform(values)
                    if isinstance(dim, int):
                        pca = PCA(n_components=1, random_state=self.random_state)
                        embedding[:, dim] = pca.fit_transform(values).flatten()
                        identity_used_dims.add(dim)
                    else:
                        poly = PolynomialFeatures(degree=len(dim))
                        transformed = poly.fit_transform(values)
                        for i, d in enumerate(dim):
                            embedding[:, d] = transformed[:, i]
                            identity_used_dims.add(d)
                    if self.verbose > 1:
                        print(f"Reducing {attr} to dimension {dim} using PCA")
            
            intersection_used_dims = set()
            # Process intersection mapping
            if intersection is not None:
                for attrs, dims_tuple in intersection.items():
                    values = self._get_values(fdf, attrs)
                    values = scaler.fit_transform(values)
                    if isinstance(dims_tuple, int):
                        if dims_tuple in identity_used_dims:
                            if self.verbose > 1:
                                print(f"Dimension {dims_tuple} of attr {attrs} for intersection is used in identity, skipping")
                        else:
                            pca = PCA(n_components=1, random_state=self.random_state)
                            embedding[:, dims_tuple] = pca.fit_transform(values).flatten()
                            intersection_used_dims.add(dims_tuple)
                            if self.verbose > 1:
                                print(f"Reducing {attrs} to dimension {dims_tuple} using PCA")
                    else:
                        if any(d in identity_used_dims for d in dims_tuple):
                            if self.verbose > 1:
                                print(f"Dimension {dims_tuple} of attr {attrs} for intersection is used in identity, skipping")
                        else:
                            pca = PCA(n_components=len(dims_tuple), random_state=self.random_state)
                            transformed = pca.fit_transform(values)
                            for i, dim in enumerate(dims_tuple):
                                embedding[:, dim] = transformed[:, i]
                                intersection_used_dims.add(dim)
                            if self.verbose > 1:
                                print(f"Reducing {attrs} to dimensions {dims_tuple} using PCA")
            
            # Process union mapping
            if union is not None:
                for attrs, dims_tuple in union.items():
                    values = self._get_values(fdf, attrs)
                    values = scaler.fit_transform(values)
                    if isinstance(dims_tuple, int):
                        if dims_tuple in identity_used_dims:
                            if self.verbose > 1:
                                print(f"Dimension {dims_tuple} of attr {attrs} for union is used in identity, skipping")
                        elif dims_tuple in intersection_used_dims:
                            if self.verbose > 1:
                                print(f"Dimension {dims_tuple} of attr {attrs} for union is used in intersection, convert it to random initialization")
                            embedding[:, dims_tuple] = 1e-4 * np.random.randn(n_samples)
                        else:
                            pca = PCA(n_components=1, random_state=self.random_state)
                            embedding[:, dims_tuple] = pca.fit_transform(values).flatten()
                            if self.verbose > 1:
                                print(f"Reducing {attrs} to dimension {dims_tuple} using PCA")
                    else:
                        if any(d in identity_used_dims for d in dims_tuple):
                            if self.verbose > 1:
                                print(f"Dimension {dims_tuple} of attr {attrs} for union is used in identity, skipping")
                        elif any(d in intersection_used_dims for d in dims_tuple):
                            if self.verbose > 1:
                                print(f"Dimension {dims_tuple} of attr {attrs} for union is used in intersection, convert it to random initialization")
                            for dim in dims_tuple:
                                embedding[:, dim] = 1e-4 * np.random.randn(n_samples)
                        else:
                            pca = PCA(n_components=len(dims_tuple), random_state=self.random_state)
                            transformed = pca.fit_transform(values)
                            for i, dim in enumerate(dims_tuple):
                                embedding[:, dim] = transformed[:, i]
                            if self.verbose > 1:
                                print(f"Reducing {attrs} to dimensions {dims_tuple} using PCA")
        elif isinstance(self.init, np.ndarray):
            assert self.init.ndim == 2, "Initial embedding must be a 2D array"
            assert self.init.shape[0] == n_samples, "Initial embedding must have the same number of samples as the input data"
            assert self.init.shape[1] == self.n_components, "Initial embedding must have the same number of components as specified"
            embedding = self.init
        elif self.init == 'random':
            pass
        else:
            raise ValueError("Init method must be 'pca', 'random', or a 2D numpy array")
        
        return embedding

    def fit_transform(self, fdf: Union[FlowDataFrame, dict],
                      identity: dict = None, intersection: dict = None, union: dict = None,
                      metrics: dict = None, relation: str = 'probability', y=None):
        """Fit the flow data into an embedded space and return the embedding.

        This is the main entry point for the ft-SNE algorithm. It validates all
        parameters, computes the high-dimensional probability distributions from
        flow attributes, and optimizes the low-dimensional embedding.

        Parameters
        ----------
        fdf : Union[FlowDataFrame, dict]
            Input flow data. If FlowDataFrame, must have valid columns or
            flow attributes ('o', 'd', 'length', 'angle'). If dict, must contain
            2D numpy arrays as values with the same number of rows and columns.
        identity : dict, optional
            Identity mapping that maps single features to embedding dimensions.
            Each key is an attribute name, and its value is the target dimension
            index (int) or tuple of indices for polynomial features.
            Example: {'o': 0, 'd': 1} maps origin to dim 0 and destination to dim 1.
        intersection : dict, optional
            Intersection mapping that models joint probability of multiple features.
            Each key is a tuple of attribute names, and its value is the target
            dimension(s).
            Example: {('o', 'd'): 0} models joint origin-destination probability
            on dimension 0.
        union : dict, optional
            Union mapping that aggregates multiple features using minimum distance.
            Each key is a tuple of attribute names, and its value is the target
            dimension(s).
            Example: {('o', 'd'): (0, 1)} maps OD union to dimensions 0 and 1.
        metrics : dict, optional
            The metric to use for each attribute's distance computation. Keys are
            attribute names, values are valid metric strings.
            If None, uses the metric defined in __init__ for all attributes.
            Example: {'o': 'euclidean', 'd': 'haversine'}
        relation : {'probability', 'distance'}, default='probability'
            Method to calculate the intersection/union probability:

            - 'probability': Use perplexity-based Gaussian kernels (recommended)
            - 'distance': Use raw distance functions (max for intersection,
              min for union)

        y : None
            Ignored. Exists for API compatibility with sklearn.

        Returns
        -------
        embedding : np.ndarray
            Embedding of the training data in low-dimensional space.
            Shape is (n_samples, n_components) where n_components is determined
            from the maximum dimension specified in mappings.

        Raises
        ------
        ValueError
            - If no mapping (identity, intersection, union) is specified
            - If relation is not 'probability' or 'distance'
            - If 'barnes_hut' method is used with intersection/union mappings
            - If 'distance' relation is used with 'barnes_hut' method
        NotImplementedError
            - If 'barnes_hut' method is used with intersection or union mappings

        Warns
        -----
        RuntimeWarning
            - If 'hd' loss function is requested with 'barnes_hut' method,
              as HD is not supported; KL divergence is used instead.

        Examples
        --------
        >>> from geoflowkit import FlowDataFrame
        >>> from geoflowkit.manifold import FTSNE

        Basic usage with identity mapping:

        >>> fdf = FlowDataFrame.from_path("flows.csv", origin="origin", destination="dest")
        >>> ft = FTSNE(perplexity=30, random_state=42, verbose=1)
        >>> embedding = ft.fit_transform(fdf, identity={'o': 0, 'd': 1})

        Using intersection mapping for flow clustering:

        >>> ft = FTSNE(perplexity=50, loss_func='hd')
        >>> embedding = ft.fit_transform(
        ...     fdf,
        ...     identity={'length': 0, 'angle': 1},
        ...     intersection={('o', 'd'): 2}
        ... )

        Using custom metrics:

        >>> embedding = ft.fit_transform(
        ...     fdf,
        ...     identity={'o': 0},
        ...     metrics={'o': 'haversine'}
        ... )
        """
        # Check parameters and get embedding dimension
        if identity is None and intersection is None and union is None:
            raise ValueError("At least one mapping (identity, intersection and union) must be specified")
        if relation not in ['probability', 'distance']:
            raise ValueError("Relation must be 'probability' or 'distance'")
        if self.method=='barnes_hut' and (intersection or union):
            raise NotImplementedError("Barnes-Hut method is not implemented yet for intersection and union mappings")
        if relation=='distance' and self.method=='barnes_hut' and (intersection or union):
            raise ValueError("Relation 'distance' is not supported for intersection and union mappings when method is 'barnes_hut'")
        if self.method=='barnes_hut' and self.loss_func=='hd':
            warnings.warn("Loss function 'hd' is not supported for Barnes-Hut method, using KL divergence instead")
            self.loss_func = 'kl'
        self.n_components = self._check_params(fdf, identity, intersection, union, metrics)
        
        # Initialize embedding
        embedding = self._initialize_embedding(fdf, identity, intersection, union, self.n_components)
        
        # Fit embedding
        return self._fit(fdf, embedding, identity, intersection, union, metrics, relation=relation)
    
    def _fit(self, fdf, embedding, identity, intersection, union, metrics, relation):
        """Compute probability distributions and run the optimization.

        This method handles the core ft-SNE computation:
        1. Computes pairwise distances for each flow attribute
        2. Converts distances to probability distributions using perplexity
        3. Runs gradient descent optimization to find the embedding

        Parameters
        ----------
        fdf : Union[FlowDataFrame, dict]
            Input flow data.
        embedding : np.ndarray
            Initial embedding matrix of shape (n_samples, n_components).
        identity : dict
            Identity mapping dictionary.
        intersection : dict
            Intersection mapping dictionary.
        union : dict
            Union mapping dictionary.
        metrics : dict
            Metrics dictionary for distance computation.
        relation : {'probability', 'distance'}
            Method for computing joint probabilities.

        Returns
        -------
        np.ndarray
            The optimized embedding matrix of shape (n_samples, n_components).

        Notes
        -----
        This method modifies pijs and projections lists that are passed to
        the optimization routine. The probability distributions are scaled
        by early_exaggeration during the initial optimization phase.
        """
        if self.learning_rate == "auto":
            # See issue #18018
            self.learning_rate_ = self.n_samples / self.early_exaggeration / 4
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate
        identity, intersection = identity or {}, intersection or {}
        union, metrics = union or {}, metrics or {}

        # Calculate pairwise distances
        attr_distances = {}
        if isinstance(fdf, FlowDataFrame):
            attrs = set(list(identity.keys()))
            for attr in list(list(union.keys())+list(intersection.keys())):
                if pd.api.types.is_scalar(attr):
                    attrs.add(attr)
                else:
                    attrs.update(attr)
            for attr in attrs:
                metric = metrics[attr] if attr in metrics else self.metric
                values = self._get_values(fdf, attr)
                if metric == "cosine":
                    if values.shape[1]==1:
                        values = np.concatenate([np.cos(values), np.sin(values)], axis=1) 
                if metric == "haversine":
                    assert values.shape[1]==2, "Haversine distance only works for 2D data"
                    if fdf.crs.is_geographic:
                        values = np.radians(values)
                    else:
                        raise ValueError("Haversine distance only works for geographic data")
                
                if self.method == "exact":
                    if metric == "euclidean":
                        distances = pairwise_distances(values, metric=metric, squared=True)
                    else:
                        metric_params_ = self.metric_params or {}
                        distances = pairwise_distances(values, metric=metric, **metric_params_)
                        distances = distances ** 2
                else:
                    n_neighbors = min(self.n_samples - 1, int(3.0 * self.perplexity + 1))
                    # Find the nearest neighbors for every point
                    knn = NearestNeighbors(
                        algorithm="auto",
                        n_jobs=self.n_jobs,
                        n_neighbors=n_neighbors,
                        metric=self.metric,
                        metric_params=self.metric_params,
                    )
                    knn.fit(values)
                    distances = knn.kneighbors_graph(mode="distance")
                    del knn
                    distances.data **= 2
                attr_distances[attr] = distances
        elif isinstance(fdf, dict):
            attr_distances = fdf

        pijs = []
        projections = []
        temp_pijs = {}
        for attrs, dims in intersection.items():
            attrs_distances = [attr_distances[attr] for attr in attrs]
            if relation == 'probability':
                if self.method == "exact":
                    attrs_p_sigma = [calc_optimized_p_cond(distances, self.perplexity) for distances in attrs_distances]
                else:
                    attrs_p_sigma = [calc_optimized_p_cond_nn(distances, self.perplexity) for distances in attrs_distances]
                attrs_sq_sigmas = [sigma for p, sigma in attrs_p_sigma]
                attrs_p = [p for p, sigma in attrs_p_sigma]
                temp_pijs.update(zip(attrs, attrs_p))
                if self.method == "exact":
                    pij = get_multivariate_p_cond(attrs_distances, attrs_sq_sigmas, combination='intersection')
                    pij = squareform(pij)
                else:
                    pij = get_multivariate_p_cond_nn(attrs_distances, attrs_sq_sigmas, combination='intersection')
            else:
                distances = np.concatenate([distances[..., np.newaxis] for distances in attrs_distances], axis=2)
                distances = np.max(distances, axis=2)
                pij = _joint_probabilities(distances, self.perplexity, 0)
            pijs.append(pij)
            projections.append(dims)

        for attrs, dims in union.items():
            attrs_distances = [attr_distances[attr] for attr in attrs]
            if relation == 'probability':
                if self.method == "exact":
                    attrs_p_sigma = [calc_optimized_p_cond(distances, self.perplexity) for distances in attrs_distances]
                else:
                    attrs_p_sigma = [calc_optimized_p_cond_nn(distances, self.perplexity) for distances in attrs_distances]
                attrs_sq_sigmas = [sigma for p, sigma in attrs_p_sigma]
                attrs_p = [p for p, sigma in attrs_p_sigma]
                temp_pijs.update(zip(attrs, attrs_p))
                if self.method == "exact":
                    pij = get_multivariate_p_cond(attrs_distances, attrs_sq_sigmas, combination='union')
                    pij = squareform(pij)
                else:
                    pij = get_multivariate_p_cond_nn(attrs_distances, attrs_sq_sigmas, combination='union')
            else:
                distances = np.concatenate([distances[..., np.newaxis] for distances in attrs_distances], axis=2)
                distances = np.min(distances, axis=2)
                pij = _joint_probabilities(distances, self.perplexity, 0)
            pijs.append(pij)
            projections.append(dims)

        for attr, dim in identity.items():
            distances = attr_distances[attr]
            if dim in temp_pijs:
                pij = temp_pijs[attr]
            else:
                if self.method == "exact":
                    pij = _joint_probabilities(distances, self.perplexity, 0)
                else:
                    pij = _joint_probabilities_nn(distances, self.perplexity, 0)
            pijs.append(pij)
            projections.append(dim)

        return self._ft_sne(embedding, pijs, projections)

    def _ft_sne(self, embedding, pijs, projections):
        """Optimize the embedding using gradient descent.

        Performs the two-phase optimization schedule:
        1. Early exaggeration phase: Higher learning rate with momentum 0.5
        2. Main optimization phase: Standard learning rate with momentum 0.8

        Parameters
        ----------
        embedding : np.ndarray
            Initial embedding matrix of shape (n_samples, n_components).
        pijs : list of np.ndarray
            List of high-dimensional probability distributions. Each pij
            corresponds to one mapping (identity, intersection, or union).
        projections : list
            List of dimension indices/tuples corresponding to each pij,
            specifying which embedding dimensions each pij projects to.

        Returns
        -------
        np.ndarray
            The optimized embedding matrix of shape (n_samples, n_components).

        Attributes
        ----------
        error_ : float
            The final optimization error (KL divergence or Hellinger distance).

        Notes
        -----
        The optimization uses a momentum-based gradient descent with two phases:

        Phase 1 (early exaggeration):
        - Duration: early_exaggeration_iter iterations
        - Learning rate: self.learning_rate_
        - Momentum: 0.5
        - Pijs scaled by early_exaggeration factor

        Phase 2 (main optimization):
        - Duration: max_iter - early_exaggeration_iter iterations
        - Learning rate: self.learning_rate_
        - Momentum: 0.8
        - Pijs at normal scale
        """
        if self.loss_func == 'kl':
            obj_func = kl_grad if self.method=='exact' else kl_grad_bh
        elif self.loss_func == 'hd':
            obj_func = hd_grad
        else:
            raise ValueError("Loss function must be 'kl' or 'hd'. ")
        degrees_of_freedom = max(self.n_components - 1, 1)
        
        self._init_pbar(self.max_iter) if self.verbose > 0 else None
        kwargs = {'angle': self.angle, 'num_threads': _openmp_effective_n_threads()}
        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        pijs = [pij * self.early_exaggeration for pij in pijs]
        optimizer = GDOptimizer(self.learning_rate_, 0.5)
        for epoch in range(self.early_exaggeration_iter):
            embedding, error = optimizer(obj_func, embedding, pijs, projections, degrees_of_freedom, **kwargs)
            if self.verbose > 0 and self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_description(f"Epoch [{epoch+1}/{self.max_iter}] KL Divergence: {error:.4f}")

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        pijs = [pij / self.early_exaggeration for pij in pijs]
        optimizer = GDOptimizer(self.learning_rate_, 0.8, lr_scheduler=None)
        for epoch in range(self.early_exaggeration_iter, self.max_iter):
            embedding, error = optimizer(obj_func, embedding, pijs, projections, degrees_of_freedom, **kwargs)
            if self.verbose > 0 and self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_description(f"Epoch [{epoch+1}/{self.max_iter}] KL Divergence: {error:.4f}")

        X_embedded = embedding.reshape(self.n_samples, self.n_components)
        self.error_ = error
        return X_embedded



