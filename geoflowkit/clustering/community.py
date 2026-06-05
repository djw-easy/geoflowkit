"""Community detection algorithms for flow data using networkx."""

import numpy as np
import networkx as nx
from networkx.algorithms.community import (
    greedy_modularity_communities,
    louvain_communities,
    modularity,
)

from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.clustering._graph_utils import (
    flows_to_graph,
    assign_flow_labels,
)


class CNMFlow:
    """Clauset-Newman-Moore greedy modularity optimization for flow data.

    Builds a flow network graph from the FlowDataFrame and detects communities
    using the CNM greedy modularity maximization algorithm.

    Parameters
    ----------
    zone_method : str, default='grid'
        Zone assignment method:
        - 'grid': Divide space into a regular grid
        - 'aggregate': Aggregate unique OD coordinate pairs
        - 'gdf': Use an external GeoDataFrame of zone polygons
        - 'custom': Use a custom zone assignment function
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type: 'count', 'volume', 'length'.
    resolution : float, default=1.0
        Modularity resolution parameter. Values > 1 produce more communities.
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').

    Attributes
    ----------
    labels_ : np.ndarray
        Community label for each flow. -1 indicates cross-community flow.
    zone_labels_ : dict
        Mapping from zone ID to community label.
    n_communities_ : int
        Number of communities found.
    modularity_ : float
        Modularity value of the partition.
    graph_ : nx.Graph
        The constructed flow network graph.
    o_zones_ : np.ndarray
        Origin zone ID for each flow.
    d_zones_ : np.ndarray
        Destination zone ID for each flow.

    References
    ----------
    [1] Clauset, A., Newman, M. E. J., & Moore, C. (2004).
        Finding community structure in very large networks.
        Physical Review E, 70(6), 066111.

    Examples
    --------
    >>> from geoflowkit import FlowDataFrame, read_csv
    >>> from geoflowkit.clustering import CNMFlow
    >>> fdf = read_csv('flows.csv', use_cols=['ox', 'oy', 'dx', 'dy'])
    >>> model = CNMFlow(zone_method='grid', cell_size=1000)
    >>> model.fit(fdf)
    >>> print(model.labels_)
    >>> print(model.n_communities_)
    """

    def __init__(self, zone_method='grid', cell_size=None,
                 weight='count', resolution=1.0, zones=None, zone_func=None):
        self.zone_method = zone_method
        self.cell_size = cell_size
        self.weight = weight
        self.resolution = resolution
        self.zones = zones
        self.zone_func = zone_func
        self.labels_ = None
        self.zone_labels_ = None
        self.n_communities_ = None
        self.modularity_ = None
        self.graph_ = None
        self.o_zones_ = None
        self.d_zones_ = None

    def fit(self, fdf):
        """Fit the CNM model to flow data.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        self
            Fitted estimator.
        """
        G, o_zones, d_zones, zone_centroids = flows_to_graph(
            fdf,
            zone_method=self.zone_method,
            cell_size=self.cell_size,
            weight=self.weight,
            zones=self.zones,
            zone_func=self.zone_func,
        )

        self.graph_ = G
        self.o_zones_ = o_zones
        self.d_zones_ = d_zones

        if len(G.nodes) == 0:
            self.labels_ = np.full(len(fdf), -1, dtype=int)
            self.zone_labels_ = {}
            self.n_communities_ = 0
            self.modularity_ = 0.0
            return self

        communities_list = greedy_modularity_communities(
            G, weight='weight', resolution=self.resolution
        )

        self.zone_labels_ = {}
        for comm_idx, community in enumerate(communities_list):
            for node in community:
                self.zone_labels_[node] = comm_idx

        self.n_communities_ = len(communities_list)
        self.modularity_ = modularity(G, communities_list, weight='weight')
        self.labels_ = assign_flow_labels(fdf, o_zones, d_zones, self.zone_labels_)

        return self

    def fit_predict(self, fdf):
        """Fit the model and return flow community labels.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        np.ndarray
            Community labels for each flow. -1 indicates cross-community flow.
        """
        self.fit(fdf)
        return self.labels_


def cnm(fdf, zone_method='grid', cell_size=None, weight='count',
        resolution=1.0, zones=None, zone_func=None):
    """Perform CNM community detection on flow data.

    This is a convenience function that creates a CNMFlow instance,
    fits it, and returns the flow-level community labels.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe.
    zone_method : str, default='grid'
        Zone assignment method: 'grid', 'aggregate', 'gdf', 'custom'.
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type: 'count', 'volume', 'length'.
    resolution : float, default=1.0
        Modularity resolution parameter.
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').

    Returns
    -------
    np.ndarray
        Community labels for each flow. -1 indicates cross-community flow.

    Examples
    --------
    >>> from geoflowkit import read_csv
    >>> from geoflowkit.clustering import cnm
    >>> fdf = read_csv('flows.csv', use_cols=['ox', 'oy', 'dx', 'dy'])
    >>> labels = cnm(fdf, zone_method='grid', cell_size=1000)
    """
    model = CNMFlow(
        zone_method=zone_method,
        cell_size=cell_size,
        weight=weight,
        resolution=resolution,
        zones=zones,
        zone_func=zone_func,
    )
    return model.fit_predict(fdf)


class LouvainFlow:
    """Louvain fast unfolding algorithm for flow data.

    Builds a flow network graph from the FlowDataFrame and detects communities
    using the Louvain modularity optimization algorithm.

    Parameters
    ----------
    zone_method : str, default='grid'
        Zone assignment method:
        - 'grid': Divide space into a regular grid
        - 'aggregate': Aggregate unique OD coordinate pairs
        - 'gdf': Use an external GeoDataFrame of zone polygons
        - 'custom': Use a custom zone assignment function
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type: 'count', 'volume', 'length'.
    resolution : float, default=1.0
        Modularity resolution parameter. Values > 1 produce more communities.
    seed : int, optional
        Random seed for reproducibility.
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').

    Attributes
    ----------
    labels_ : np.ndarray
        Community label for each flow. -1 indicates cross-community flow.
    zone_labels_ : dict
        Mapping from zone ID to community label.
    n_communities_ : int
        Number of communities found.
    modularity_ : float
        Modularity value of the partition.
    graph_ : nx.Graph
        The constructed flow network graph.
    o_zones_ : np.ndarray
        Origin zone ID for each flow.
    d_zones_ : np.ndarray
        Destination zone ID for each flow.

    References
    ----------
    [1] Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
        Fast unfolding of communities in large networks.
        Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.

    Examples
    --------
    >>> from geoflowkit import read_csv
    >>> from geoflowkit.clustering import LouvainFlow
    >>> fdf = read_csv('flows.csv', use_cols=['ox', 'oy', 'dx', 'dy'])
    >>> model = LouvainFlow(zone_method='grid', cell_size=1000)
    >>> model.fit(fdf)
    >>> print(model.labels_)
    >>> print(model.n_communities_)
    """

    def __init__(self, zone_method='grid', cell_size=None,
                 weight='count', resolution=1.0, seed=None,
                 zones=None, zone_func=None):
        self.zone_method = zone_method
        self.cell_size = cell_size
        self.weight = weight
        self.resolution = resolution
        self.seed = seed
        self.zones = zones
        self.zone_func = zone_func
        self.labels_ = None
        self.zone_labels_ = None
        self.n_communities_ = None
        self.modularity_ = None
        self.graph_ = None
        self.o_zones_ = None
        self.d_zones_ = None

    def fit(self, fdf):
        """Fit the Louvain model to flow data.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        self
            Fitted estimator.
        """
        G, o_zones, d_zones, zone_centroids = flows_to_graph(
            fdf,
            zone_method=self.zone_method,
            cell_size=self.cell_size,
            weight=self.weight,
            zones=self.zones,
            zone_func=self.zone_func,
        )

        self.graph_ = G
        self.o_zones_ = o_zones
        self.d_zones_ = d_zones

        if len(G.nodes) == 0:
            self.labels_ = np.full(len(fdf), -1, dtype=int)
            self.zone_labels_ = {}
            self.n_communities_ = 0
            self.modularity_ = 0.0
            return self

        communities_list = louvain_communities(
            G, weight='weight', resolution=self.resolution, seed=self.seed
        )

        self.zone_labels_ = {}
        for comm_idx, community in enumerate(communities_list):
            for node in community:
                self.zone_labels_[node] = comm_idx

        self.n_communities_ = len(communities_list)
        self.modularity_ = modularity(G, communities_list, weight='weight')
        self.labels_ = assign_flow_labels(fdf, o_zones, d_zones, self.zone_labels_)

        return self

    def fit_predict(self, fdf):
        """Fit the model and return flow community labels.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        np.ndarray
            Community labels for each flow. -1 indicates cross-community flow.
        """
        self.fit(fdf)
        return self.labels_


def louvain(fdf, zone_method='grid', cell_size=None, weight='count',
            resolution=1.0, seed=None, zones=None, zone_func=None):
    """Perform Louvain community detection on flow data.

    This is a convenience function that creates a LouvainFlow instance,
    fits it, and returns the flow-level community labels.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe.
    zone_method : str, default='grid'
        Zone assignment method: 'grid', 'aggregate', 'gdf', 'custom'.
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type: 'count', 'volume', 'length'.
    resolution : float, default=1.0
        Modularity resolution parameter.
    seed : int, optional
        Random seed for reproducibility.
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').

    Returns
    -------
    np.ndarray
        Community labels for each flow. -1 indicates cross-community flow.

    Examples
    --------
    >>> from geoflowkit import read_csv
    >>> from geoflowkit.clustering import louvain
    >>> fdf = read_csv('flows.csv', use_cols=['ox', 'oy', 'dx', 'dy'])
    >>> labels = louvain(fdf, zone_method='grid', cell_size=1000)
    """
    model = LouvainFlow(
        zone_method=zone_method,
        cell_size=cell_size,
        weight=weight,
        resolution=resolution,
        seed=seed,
        zones=zones,
        zone_func=zone_func,
    )
    return model.fit_predict(fdf)


class STOCSFlow:
    """Spatial Tabu Optimization for Community Structure in flow data.

    Builds a flow network graph and detects communities using a tabu search
    algorithm that optimizes modularity with spatial proximity constraints.
    Ensures that nodes within the same community are spatially close.

    Parameters
    ----------
    zone_method : str, default='grid'
        Zone assignment method:
        - 'grid': Divide space into a regular grid
        - 'aggregate': Aggregate unique OD coordinate pairs
        - 'gdf': Use an external GeoDataFrame of zone polygons
        - 'custom': Use a custom zone assignment function
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type: 'count', 'volume', 'length'.
    resolution : float, default=1.0
        Modularity resolution parameter.
    spatial_weight : float, default=0.5
        Weight for spatial proximity constraint in the objective function.
        Range [0, 1]: 0 = pure modularity, 1 = pure spatial proximity.
    tabu_tenure : int, default=10
        Tabu list tenure (number of iterations a move is forbidden).
    max_iter : int, default=100
        Maximum number of iterations.
    n_init : int, default=5
        Number of random restarts.
    seed : int, optional
        Random seed for reproducibility.
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').

    Attributes
    ----------
    labels_ : np.ndarray
        Community label for each flow. -1 indicates cross-community flow.
    zone_labels_ : dict
        Mapping from zone ID to community label.
    n_communities_ : int
        Number of communities found.
    modularity_ : float
        Modularity value of the partition.
    graph_ : nx.Graph
        The constructed flow network graph.
    o_zones_ : np.ndarray
        Origin zone ID for each flow.
    d_zones_ : np.ndarray
        Destination zone ID for each flow.

    References
    ----------
    [1] Guo, D., Jin, M., & Zhu, S. (2018).
        Detecting spatial community structure in movements.
        International Journal of Geographical Information Science, 32(7): 1326-1347.

    Examples
    --------
    >>> from geoflowkit import read_csv
    >>> from geoflowkit.clustering import STOCSFlow
    >>> fdf = read_csv('flows.csv', use_cols=['ox', 'oy', 'dx', 'dy'])
    >>> model = STOCSFlow(zone_method='grid', cell_size=1000,
    ...                   spatial_weight=0.5, tabu_tenure=15)
    >>> model.fit(fdf)
    >>> print(model.labels_)
    """

    def __init__(self, zone_method='grid', cell_size=None,
                 weight='count', resolution=1.0, spatial_weight=0.5,
                 tabu_tenure=10, max_iter=100, n_init=5, seed=None,
                 zones=None, zone_func=None):
        self.zone_method = zone_method
        self.cell_size = cell_size
        self.weight = weight
        self.resolution = resolution
        self.spatial_weight = spatial_weight
        self.tabu_tenure = tabu_tenure
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        self.zones = zones
        self.zone_func = zone_func
        self.labels_ = None
        self.zone_labels_ = None
        self.n_communities_ = None
        self.modularity_ = None
        self.graph_ = None
        self.o_zones_ = None
        self.d_zones_ = None

    def _compute_spatial_distance(self, zone_centroids):
        """Compute pairwise spatial distance matrix between zones.

        Parameters
        ----------
        zone_centroids : dict
            Mapping from zone ID to (x, y) centroid.

        Returns
        -------
        dist_matrix : dict
            Nested dict: dist_matrix[i][j] = Euclidean distance between zones i and j.
        max_dist : float
            Maximum distance between any two zones.
        """
        zone_ids = sorted(zone_centroids.keys())
        dist_matrix = {}
        max_dist = 0.0

        for i in zone_ids:
            dist_matrix[i] = {}
            for j in zone_ids:
                if i == j:
                    dist_matrix[i][j] = 0.0
                elif j in dist_matrix and i in dist_matrix[j]:
                    dist_matrix[i][j] = dist_matrix[j][i]
                else:
                    xi, yi = zone_centroids[i]
                    xj, yj = zone_centroids[j]
                    d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    dist_matrix[i][j] = d
                    if d > max_dist:
                        max_dist = d

        return dist_matrix, max_dist

    def _compute_spatial_penalty(self, node, new_comm, zone_labels, zone_centroids,
                                 dist_matrix, max_dist):
        """Compute spatial penalty for moving a node to a new community.

        Parameters
        ----------
        node : int
            Zone node to evaluate.
        new_comm : int
            Target community label.
        zone_labels : dict
            Current zone-to-community mapping.
        zone_centroids : dict
            Zone centroids.
        dist_matrix : dict
            Pairwise distance matrix.
        max_dist : float
            Maximum distance.

        Returns
        -------
        penalty : float
            Spatial penalty in [0, 1]. Higher = worse spatial cohesion.
        """
        same_comm_nodes = [n for n, c in zone_labels.items()
                          if c == new_comm and n != node]

        if len(same_comm_nodes) == 0:
            return 0.0

        total_dist = sum(dist_matrix[node][n] for n in same_comm_nodes)
        avg_dist = total_dist / len(same_comm_nodes)

        if max_dist > 0:
            return avg_dist / max_dist
        return 0.0

    def _objective(self, G, zone_labels, zone_centroids, dist_matrix, max_dist):
        """Compute combined objective: modularity + spatial penalty.

        Parameters
        ----------
        G : nx.Graph
            Flow network graph.
        zone_labels : dict
            Zone-to-community mapping.
        zone_centroids : dict
            Zone centroids.
        dist_matrix : dict
            Pairwise distance matrix.
        max_dist : float
            Maximum distance.

        Returns
        -------
        obj : float
            Objective value (higher is better).
        """
        communities = {}
        for node, comm in zone_labels.items():
            if comm not in communities:
                communities[comm] = set()
            communities[comm].add(node)

        community_list = list(communities.values())
        if len(community_list) == 0:
            return -np.inf

        mod = modularity(G, community_list, weight='weight')

        spatial_penalties = []
        for comm_nodes in community_list:
            if len(comm_nodes) <= 1:
                continue
            dists = []
            for i in comm_nodes:
                for j in comm_nodes:
                    if i < j:
                        dists.append(dist_matrix[i][j])
            if dists and max_dist > 0:
                spatial_penalties.append(np.mean(dists) / max_dist)

        avg_spatial = np.mean(spatial_penalties) if spatial_penalties else 0.0

        return mod - self.spatial_weight * avg_spatial

    def _fit_single(self, G, zone_centroids, rng):
        """Single STOCS optimization run.

        Parameters
        ----------
        G : nx.Graph
            Flow network graph.
        zone_centroids : dict
            Zone centroids.
        rng : numpy.random.RandomState
            Random number generator.

        Returns
        -------
        best_labels : dict
            Best zone-to-community mapping found.
        best_obj : float
            Best objective value.
        """
        nodes = list(G.nodes)
        n_nodes = len(nodes)

        zone_labels = {node: i for i, node in enumerate(nodes)}

        dist_matrix, max_dist = self._compute_spatial_distance(zone_centroids)

        best_labels = dict(zone_labels)
        best_obj = self._objective(G, zone_labels, zone_centroids, dist_matrix, max_dist)

        tabu_list = {}

        for iteration in range(self.max_iter):
            node_to_move = nodes[rng.randint(0, n_nodes)]
            current_comm = zone_labels[node_to_move]

            neighbors = set()
            for neighbor in G.neighbors(node_to_move):
                neighbors.add(zone_labels[neighbor])

            candidate_comms = list(neighbors)
            if current_comm not in candidate_comms:
                candidate_comms.append(current_comm)

            best_move_comm = None
            best_move_obj = -np.inf

            for comm in candidate_comms:
                if comm == current_comm:
                    continue

                is_tabu = (node_to_move in tabu_list and
                          tabu_list[node_to_move] > iteration and
                          comm != current_comm)

                old_labels = dict(zone_labels)
                zone_labels[node_to_move] = comm
                obj = self._objective(G, zone_labels, zone_centroids, dist_matrix, max_dist)
                zone_labels[node_to_move] = current_comm

                if not is_tabu or obj > best_obj:
                    if obj > best_move_obj:
                        best_move_obj = obj
                        best_move_comm = comm

            if best_move_comm is not None:
                zone_labels[node_to_move] = best_move_comm
                tabu_list[node_to_move] = iteration + self.tabu_tenure

                if best_move_obj > best_obj:
                    best_obj = best_move_obj
                    best_labels = dict(zone_labels)

        return best_labels, best_obj

    def fit(self, fdf):
        """Fit the STOCS model to flow data.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        self
            Fitted estimator.
        """
        G, o_zones, d_zones, zone_centroids = flows_to_graph(
            fdf,
            zone_method=self.zone_method,
            cell_size=self.cell_size,
            weight=self.weight,
            zones=self.zones,
            zone_func=self.zone_func,
        )

        self.graph_ = G
        self.o_zones_ = o_zones
        self.d_zones_ = d_zones

        if len(G.nodes) == 0:
            self.labels_ = np.full(len(fdf), -1, dtype=int)
            self.zone_labels_ = {}
            self.n_communities_ = 0
            self.modularity_ = 0.0
            return self

        rng = np.random.RandomState(self.seed)

        best_zone_labels = None
        best_obj = -np.inf

        for _ in range(self.n_init):
            zone_labels, obj = self._fit_single(G, zone_centroids, rng)
            if obj > best_obj:
                best_obj = obj
                best_zone_labels = dict(zone_labels)

        self.zone_labels_ = best_zone_labels

        communities = {}
        for node, comm in best_zone_labels.items():
            if comm not in communities:
                communities[comm] = set()
            communities[comm].add(node)

        community_list = list(communities.values())
        self.n_communities_ = len(community_list)
        self.modularity_ = modularity(G, community_list, weight='weight')
        self.labels_ = assign_flow_labels(fdf, o_zones, d_zones, self.zone_labels_)

        return self

    def fit_predict(self, fdf):
        """Fit the model and return flow community labels.

        Parameters
        ----------
        fdf : FlowDataFrame
            The input flow dataframe.

        Returns
        -------
        np.ndarray
            Community labels for each flow. -1 indicates cross-community flow.
        """
        self.fit(fdf)
        return self.labels_


def stocs(fdf, zone_method='grid', cell_size=None, weight='count',
          resolution=1.0, spatial_weight=0.5, tabu_tenure=10,
          max_iter=100, n_init=5, seed=None, zones=None, zone_func=None):
    """Perform STOCS community detection on flow data.

    This is a convenience function that creates a STOCSFlow instance,
    fits it, and returns the flow-level community labels.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe.
    zone_method : str, default='grid'
        Zone assignment method: 'grid', 'aggregate', 'gdf', 'custom'.
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type: 'count', 'volume', 'length'.
    resolution : float, default=1.0
        Modularity resolution parameter.
    spatial_weight : float, default=0.5
        Weight for spatial proximity constraint.
    tabu_tenure : int, default=10
        Tabu list tenure.
    max_iter : int, default=100
        Maximum number of iterations.
    n_init : int, default=5
        Number of random restarts.
    seed : int, optional
        Random seed for reproducibility.
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').

    Returns
    -------
    np.ndarray
        Community labels for each flow. -1 indicates cross-community flow.

    Examples
    --------
    >>> from geoflowkit import read_csv
    >>> from geoflowkit.clustering import stocs
    >>> fdf = read_csv('flows.csv', use_cols=['ox', 'oy', 'dx', 'dy'])
    >>> labels = stocs(fdf, zone_method='grid', cell_size=1000,
    ...               spatial_weight=0.5, tabu_tenure=15)
    """
    model = STOCSFlow(
        zone_method=zone_method,
        cell_size=cell_size,
        weight=weight,
        resolution=resolution,
        spatial_weight=spatial_weight,
        tabu_tenure=tabu_tenure,
        max_iter=max_iter,
        n_init=n_init,
        seed=seed,
        zones=zones,
        zone_func=zone_func,
    )
    return model.fit_predict(fdf)
