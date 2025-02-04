import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


from shapely import get_coordinates
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold._t_sne import _joint_probabilities


from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.utils.ftsne.utils import (
    calc_optimized_p_cond, 
    get_multivariate_p_cond, 
    kl_grad, hd_grad,
    GDOptimizer
)


class FTSNE:
    """ft-SNE A Variant of t-SNE for Visualizing Geographical Flow Data
    
    Parameters
    ----------
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.
    learning_rate: float (optional, default 'auto')
        The initial learning rate for the embedding optimization. 
        The 'auto' option sets the learning_rate
        to `max(N / early_exaggeration / 4, 50)` where N is the sample size, following [4] and [5].
    max_iter: int (optional, default 1000)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. 
    early_exaggeration: float (optional, default 12.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    early_exaggeration_iter: float (optional, default 'auto')
        Number of training cycles in which exaggeration will be applied. 
    init: str (optional, default 'pca')
        Method to use for initialization of the embedding. 
        Options are pca, random, or a np.ndarray of shape (n_samples, n_components)
    method : {'exact'}, default='exact'
        # TODO: Add support for 'barnes_hut'
    random_state: int (optional, default None)
        Seed for random number generator
    loss_func: str (optional, default 'kl')
        Loss function to use for optimization. Options are 'kl' or 'hd'
    metric: str (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. 
        Valid string metrics include:
        From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. 
        These metrics support sparse matrix inputs. ['nan_euclidean'] but it does not yet support sparse matrices.
        From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 
        'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 
        'sokalsneath', 'sqeuclidean', 'yule'] See the documentation for scipy.spatial.distance for details on these metrics. 
        These metrics do not support sparse matrix inputs.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    log_progress: str (optional, default 'tqdm')
        Method to use for logging progress. Options are 'tqdm' or 'notebook'
    n_jobs: int (optional, default None)
        The number of parallel jobs to run for pairwise distances calculation
    verbose : int, default=0
        Verbosity level. If non-zero, progress is printed to stdout.
    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    
    Reference:
    ---------
    [1] Dong J, Pei T*, et al. Visualizing geographical flow data using ft-SNE, International Journal of Geographical Information Science, 2025.
    """
    # Valid distance metrics
    VALID_METRICS = [
        'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
        'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
        'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'
    ]

    def __init__(self, 
                 perplexity=30.0, 
                 learning_rate=0.1,
                 max_iter=1000, 
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
        """Check parameters and calculate embedding dimension
        
        Parameters
        ----------
        fdf : Union[FlowDataFrame, dict]
            Input flow data. If dict, must contain 2D numpy arrays as values with same number of rows and columns
        identity : dict, optional
            Identity mapping
        intersection : dict, optional 
            Intersection mapping
        union : dict, optional
            Union mapping
            
        Returns
        -------
        int
            Calculated embedding dimension
            
        Raises
        ------
        ValueError
            If any parameter is invalid
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
                if metric not in self.VALID_METRICS:
                    raise ValueError(f"Invalid metric {metric}. Valid metrics are: {self.VALID_METRICS}")

        return self.n_components
    
    def _get_values(self, fdf: FlowDataFrame, attr: Union[str, tuple]):
        # 定义一个内部函数，用于从属性中获取值
        def _get_values_from_attr(attr):
            # 如果属性不在fdf的列中，且属性为'o'或'd'，则返回get_coordinates函数的返回值
            if attr not in fdf.columns and attr in ['o', 'd']:
                return get_coordinates(getattr(fdf, attr))
            # 如果属性不在fdf的列中，且属性为'length'或'angle'，则返回getattr函数的返回值，并将其重塑为(-1, 1)的形状
            elif attr not in fdf.columns and attr in ['length', 'angle']:
                return getattr(fdf, attr).values.reshape(-1, 1)
            # 如果属性在fdf的列中，则返回fdf[attr]的值，并将其重塑为(-1, 1)的形状
            elif attr in fdf.columns:
                return fdf[attr].values.reshape(-1, 1)
            else:
                raise ValueError(f"Invalid attribute: {attr}, not in {fdf.columns} or ['o', 'd', 'length', 'angle']")
        
        if pd.api.types.is_scalar(attr):
            return _get_values_from_attr(attr)
        elif pd.api.types.is_list_like(attr):
            values = []
            for attr_ in attr:
                values.append(_get_values_from_attr(attr_))
            return np.concatenate(values, axis=1)
        else:
            raise ValueError(f"Invalid attribute type {type(attr)}, must be str or tuple")
        
    def _initialize_embedding(self, fdf: Union[FlowDataFrame, dict], identity: dict, 
                            intersection: dict, union: dict, n_components: int):
        """Initialize the embedding matrix
        
        Parameters
        ----------
        fdf : Union[FlowDataFrame, dict]
            Input flow data. If dict, must contain 2D numpy arrays as values with same number of rows
        identity : dict
            Identity mapping
        intersection : dict
            Intersection mapping
        union : dict
            Union mapping
        n_components : int
            Number of embedding dimensions
            
        Returns
        -------
        np.ndarray
            Initial embedding matrix
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
        # Check parameters and get embedding dimension
        if identity is None and intersection is None and union is None:
            raise ValueError("At least one mapping (identity, intersection and union) must be specified")
        self.n_components = self._check_params(fdf, identity, intersection, union, metrics)
        
        # Initialize embedding
        embedding = self._initialize_embedding(fdf, identity, intersection, union, self.n_components)
        
        # Fit embedding
        return self._fit(fdf, embedding, identity, intersection, union, metrics, relation=relation)
    
    def _fit(self, fdf, embedding, identity, intersection, union, metrics, relation):
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
                
                if metric == "euclidean":
                    distances = pairwise_distances(values, metric=metric, squared=True)
                else:
                    metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(values, metric=metric, **metric_params_)
                    distances = distances ** 2
                # attr_distances[attr] = distances / distances.max()
                attr_distances[attr] = distances
        elif isinstance(fdf, dict):
            attr_distances = fdf

        pijs = []
        projections = []
        temp_pijs = {}
        for attrs, dims in intersection.items():
            attrs_distances = [attr_distances[attr] for attr in attrs]
            if relation == 'probability':
                attrs_p_sigma = [calc_optimized_p_cond(distances, self.perplexity) for distances in attrs_distances]
                attrs_sigmas = [sigma for p, sigma in attrs_p_sigma]
                attrs_p = [p for p, sigma in attrs_p_sigma]
                temp_pijs.update(zip(dims, attrs_p))
                pij = get_multivariate_p_cond(attrs_distances, attrs_sigmas, combination='intersection')
                pij = squareform(pij)
            else:
                distances = np.concatenate([distances[..., np.newaxis] for distances in attrs_distances], axis=2)
                distances = np.max(distances, axis=2)
                pij = _joint_probabilities(distances, self.perplexity, 0)
            pijs.append(pij)
            projections.append(dims)

        for attrs, dims in union.items():
            attrs_distances = [attr_distances[attr] for attr in attrs]
            if relation == 'probability':
                attrs_p_sigma = [calc_optimized_p_cond(distances, self.perplexity) for distances in attrs_distances]
                attrs_sigmas = [sigma for p, sigma in attrs_p_sigma]
                attrs_p = [p for p, sigma in attrs_p_sigma]
                temp_pijs.update(zip(dims, attrs_p))
                pij = get_multivariate_p_cond(attrs_distances, attrs_sigmas, combination='union')
                pij = squareform(pij)
            else:
                distances = np.concatenate([distances[..., np.newaxis] for distances in attrs_distances], axis=2)
                distances = np.min(distances, axis=2)
                pij = _joint_probabilities(distances, self.perplexity, 0)
            pijs.append(pij)
            projections.append(dims)

        for attr, dim in identity.items():
            distances = attr_distances[attr]
            if dim in temp_pijs:
                pij = temp_pijs[dim]
            else:
                pij = _joint_probabilities(distances, self.perplexity, 0)
            pijs.append(pij)
            projections.append(dim)

        return self._ft_sne(embedding, pijs, projections)

    def _ft_sne(self, embedding, pijs, projections):
        if self.loss_func == 'kl':
            obj_func = kl_grad
        elif self.loss_func == 'hd':
            obj_func = hd_grad
        else:
            raise ValueError("Loss function must be 'kl' or 'hd'. ")
        degrees_of_freedom = max(self.n_components - 1, 1)
        
        self._init_pbar(self.max_iter) if self.verbose > 0 else None
        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        pijs = [pij * self.early_exaggeration for pij in pijs]
        optimizer = GDOptimizer(self.learning_rate_, 0.5)
        for epoch in range(self.early_exaggeration_iter):
            embedding, error = optimizer(obj_func, embedding, pijs, projections, degrees_of_freedom)
            if self.verbose > 0 and self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_description(f"Epoch [{epoch+1}/{self.max_iter}] KL Divergence: {error:.4f}")

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        pijs = [pij / self.early_exaggeration for pij in pijs]
        optimizer = GDOptimizer(self.learning_rate_, 0.8, lr_scheduler=None)
        for epoch in range(self.early_exaggeration_iter, self.max_iter):
            embedding, error = optimizer(obj_func, embedding, pijs, projections, degrees_of_freedom)
            if self.verbose > 0 and self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_description(f"Epoch [{epoch+1}/{self.max_iter}] KL Divergence: {error:.4f}")

        X_embedded = embedding.reshape(self.n_samples, self.n_components)
        self.error_ = error
        return X_embedded



