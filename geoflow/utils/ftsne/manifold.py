import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_random_state
from sklearn.decomposition import PCA

class Sammon:
    """
    Sammon mapping is a dimensionality reduction technique that maps high-dimensional data to a lower-dimensional space while preserving the pairwise distances between points.
    It is a variant of the more general multidimensional scaling (MDS) technique.
    
    Args:
        n_components: The dimensionality of the target space.
        max_iter: The maximum number of iterations for the sammon mapping to try and find the optimal mapping.
        init: The initialization method for the embedding.
        metric: The metric to use for the distance calculation.
        epsilon: The cost of the mapping the function strives to get under.
        maxhalves: How many times the algorithm will try to halve the gradient step size if E doesn't decrease.
        verbose: If greater than 1 will print out E after each iteration.
        minsize: If values in the distance matrices are smaller than this value, they will be set to this value.
        random_state: The random state to use for the initialization.
        
    Reference:
        Sammon, J. W. (1969). A non-linear mapping for data structure analysis. IEEE Transactions on Computers, 18(8), 1002-1009.
        https://github.com/SkoogJacob/SammonMapping/blob/main/models/sammon.py
    """
    def __init__(self, n_components, max_iter=100, init='random', metric='euclidean', 
                 epsilon=0.0001, maxhalves=16, verbose=0, minsize=1e-10, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.init = init
        self.metric = metric
        self.epsilon = epsilon
        self.maxhalves = maxhalves
        self.verbose = verbose
        self.minsize = minsize
        self.random_state = random_state

    def fit_transform(self, x: np.ndarray, y=None):
        """
        :param x: The data to map with sammon mapping. Should be an numpy.ndarray of shape n*p, where n is the number of entries and p is the number of features.
        :return: An array of points in the new space of dimensionality q.
        """
        N = x.shape[0]
        random_state = check_random_state(self.random_state)
        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            # Always output a numpy array, no matter what is configured globally
            pca.set_output(transform="default")
            X_embedded = pca.fit_transform(x).astype(np.float32, copy=False)
            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif self.init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(N, self.n_components)
            ).astype(np.float32)
        
        D = pairwise_distances(x, metric=self.metric)  # Saving all pairwise distances between points in x. Each distance is present twice!
        # equiv to dstar in the original paper
        Dsuminv = 1 / D.sum()  # Summing the pairwise distances. Note each distance is present twice, making the sum be
        # Twice as big as it should be compared to in the paper. In terms of the paper this is
        # 1 / 2c
        D = D + np.eye(N)  # Adding the identity matrix to D as D will be used in division quite a bit where the one is a
        # neutral entity.
        D = np.where(D < self.minsize, self.minsize, D)  # Swapping values in D that are too small for minsize
        if np.count_nonzero(D) < D.size:
            print("Warning! x seems to contain identical points!")
        Dinv = 1 / D  # To avoid divisions by 0, take the inverse now and use it in multiplication later
        one = np.ones(X_embedded.shape)  # Creating an array of ones for summations later
        d = pairwise_distances(X_embedded, metric='euclidean') + np.eye(N)  # Taking pairwise distances of y. Equivalent to d in original paper. Adding identity
        d = np.where(d < self.minsize, self.minsize, d)
        # To allow for division with it
        dinv = 1 / d  # Same reasoning as for Dinv
        delta = D - d  # For sammon's stress, take distance between the distance

        # Now compute stress
        # Note that when squaring the double size problem also squares, thus dividing it further
        # by 2
        E = 0.5 * Dsuminv * ((delta ** 2) * Dinv).sum()
        current_iter = 0
        while E > self.epsilon and current_iter < self.max_iter:
            j, E_new = 0, 0.
            delta = dinv - Dinv  # This equals 1/d[i,j] - 1/D[i,j] = (D[i,j] - d[i,j]) / d[i,j]D[i,j], which is the first
            # Factor in computing the first derivative of E
            deltas = delta.dot(one)  # Creates an array of shape (N, q) where each entry
            # e[i, q] = sum[j, N, j != i] ( (D[i,j] - d[i,j]) / d[i,j]D[i,j] )
            y_diffs = np.zeros((N, self.n_components, N))
            first_derivative = -1 * (deltas * X_embedded - delta.dot(X_embedded))  # Since both first derivative and second are multiplied by
            # -2/c, and the absolute is taken of the second derivative
            # only the sign is important to keep in the first derivative
            y2 = X_embedded ** 2
            dinv3 = dinv ** 3
            # Second derivative can be written as "deltas - (1/d^3)(y[i] - y[j])^2"
            # calculating in two steps, starting by taking dinv**3 ( y[p]^2 + y[j]^2) - 2 dinv**3 y[p]y[j] and then
            # adding deltaones, end by taking absolute value
            second_derivative = dinv3.dot(one) * y2 + dinv3.dot(y2) - 2 * X_embedded * dinv3.dot(X_embedded) - deltas

            step = 0.3 * first_derivative / np.abs(second_derivative)  # step = MF * UpperCaseDelta(m)
            y_old = X_embedded

            for j in range(self.maxhalves):  # Does progressively smaller gradient steps as long as the new step didn't improve
                # stress
                X_embedded = y_old - step
                d = pairwise_distances(X_embedded, metric='euclidean') + np.eye(N)
                d = np.where(d < self.minsize, self.minsize, d)
                dinv = 1 / d
                delta = D - d
                E_new = 0.5 * Dsuminv * ((delta ** 2) * Dinv).sum()
                if E_new < E:
                    break
                else:
                    step = step / 2

            if j == self.maxhalves - 1:
                print('Warning! Steps seem to be too large, mapping may not converge')
            if E_new >= E:
                print('Warning! Mapping did not improve, returning current mapping!')
                break

            E = E_new
            current_iter = current_iter + 1
            if self.verbose > 0:
                print(f'Finished iteration {current_iter} with E = {np.around(E, decimals=4)}')
                
        if current_iter == self.max_iter:
            print("Warning! max_iter reached, mapping may not have converged.")
        
        return X_embedded


class LE:
    """
    Laplacian Eigenmaps (LE) for dimensionality reduction.
    
    Args:
        n_components: The dimensionality of the target space.
        n_neighbors: Number of nearest neighbors for the RBF kernel.
        ratio: The ratio of the maximum distance to the parameter for RBF kernel.
        
    Reference:
        Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data representation. Neural computation, 15(6), 1373-1396.
        https://github.com/heucoder/dimensionality_reduction_alo_codes/blob/master/codes/LE/LE.py
    """
    def __init__(self, n_components=2, n_neighbors=5, ratio=0.1, metric='euclidean', verbose=0):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.ratio = ratio
        self.metric = metric
        self.verbose = verbose
        
    def _rbf(self, dist):
        """RBF kernel function"""
        return np.exp(-(dist/self.t))
        
    def _cal_rbf_dist(self, data):
        """Calculate RBF kernel distances"""
        dist = pairwise_distances(data, metric=self.metric)
        self.t = np.max(dist) * self.ratio
        dist[dist < 0] = 0
        n = dist.shape[0]
        rbf_dist = self._rbf(dist)

        W = np.zeros((n, n))
        for i in range(n):
            index_ = np.argsort(dist[i])[1:1+self.n_neighbors]
            W[i, index_] = rbf_dist[i, index_]
            W[index_, i] = rbf_dist[index_, i]

        return W
        
    def fit_transform(self, data, y=None):
        """
        Fit LE model and transform data.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Transformed data in the embedded space
        """
        N = data.shape[0]
        W = self._cal_rbf_dist(data)
        D = np.zeros_like(W)
        for i in range(N):
            D[i, i] = np.sum(W[i])

        D_inv = np.linalg.inv(D)
        L = D - W
        eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))

        sort_index_ = np.argsort(eig_val)
        eig_val = eig_val[sort_index_]
        
        if self.verbose:
            print("eig_val[:10]: ", eig_val[:10])

        j = 0
        while eig_val[j] < 1e-6:
            j+=1

        if self.verbose:
            print("j: ", j)

        sort_index_ = sort_index_[j: j+self.n_components]
        eig_val_picked = eig_val[j: j+self.n_components]
        
        if self.verbose:
            print(eig_val_picked)
            print(np.dot(np.dot(eig_vec_picked.T, D), eig_vec_picked))

        eig_vec_picked = eig_vec[:, sort_index_]
        return eig_vec_picked



def flow_min_dis(flow1, flow2):
    x = (flow1 - flow2) ** 2
    o_dis = np.sqrt(np.sum(x[:2]))
    d_dis = np.sqrt(np.sum(x[2:4]))
    return np.min([o_dis, d_dis])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('E:/Research/FlowManifold/code2')
    from flow.dataset import GeoFlowDataset
    from flow.tools import map_flow_color

    root_dir = "E:/Research/FlowManifold/code2/data/sys_flow"
    dataset = "sys_flow_dataset2"
    gfd = GeoFlowDataset(file_path=f'{root_dir}/{dataset}.gpkg', extent_ratio=0.02)
    X = gfd.flow_df[['oy', 'ox', 'dy', 'dx']].values
    flow_colors = map_flow_color(gfd.flow_df['flow_type'])

    transformer = Sammon(n_components=2, max_iter=100, metric=flow_min_dis, 
                         epsilon=0.0001, maxhalves=16, verbose=1, minsize=1e-10)
    X_embedded = transformer.fit_transform(X)
    
    # transformer = LE(n_components=2, n_neighbors=200, ratio=0.1, metric=flow_min_dis)
    # X_embedded = transformer.fit_transform(X)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=flow_colors)
    plt.show()





