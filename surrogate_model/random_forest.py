import numpy as np
from pyrfr import regression

from .base_model import BaseModel


class RFR(BaseModel):
    def __init__(self,
                 num_trees: int = 10,
                 do_bootstrapping: bool = False,
                 n_points_per_tree: int = -1,
                 ratio_features: float = 1.,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_depth: int = 2 ** 20,
                 eps_purity: float = 1e-8,
                 max_num_nodes: int = 2 ** 20,
                 rng: int = 1,
                 ):
        """
        Parameters
        ----------
        num_trees : int
            The number of trees in the random forest.
        do_bootstrapping : bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree : int
            Number of points per tree. If <= 0 X.shape[0] will be used
            in _train(X, y) instead
        ratio_features : float
            The ratio of features that are considered for splitting.
        min_samples_split : int
            The minimum number of data points to perform a split.
        min_samples_leaf : int
            The minimum number of data points in a leaf.
        max_depth : int
            The maximum depth of a single tree.
        eps_purity : float
            The minimum difference between two target values to be considered
            different
        max_num_nodes : int
            The maxmimum total number of nodes in a tree
        """
        super(RFR, self).__init__()
        self.seed = rng
        self.rng = regression.default_random_engine(rng)

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.do_bootstrapping = do_bootstrapping
        self.ratio_features = ratio_features

        self.rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self.rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self.rf_opts.tree_opts.max_depth = max_depth
        self.rf_opts.tree_opts.epsilon_purity = eps_purity
        self.rf_opts.tree_opts.max_num_nodes = max_num_nodes
        self.rf_opts.compute_law_of_total_variance = False

        self.n_points_per_tree = n_points_per_tree
        self.rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, self.seed]

    def train(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestWithInstances':
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """

        self.is_trained = True

        X = X
        y = y.flatten()

        if len(X.shape) <= 1:
            num_features = 1
        else:
            num_features = np.shape(X)[-1]

        max_features = 0 if self.ratio_features > 1.0 else \
            max(1, int(num_features * self.ratio_features))
        self.rf_opts.tree_opts.max_features = max_features

        if self.n_points_per_tree <= 0:
            self.rf_opts.num_data_points_per_tree = X.shape[0]
        else:
            self.rf_opts.num_data_points_per_tree = self.n_points_per_tree

        self.rf = regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self._init_data_container(X, y)
        self.rf.fit(data, rng=self.rng)
        return self

    def _init_data_container(self, X: np.ndarray, y: np.ndarray) -> regression.default_data_container:
        """Fills a pyrfr default data container, s.t. the forest knows
        categoricals and bounds for continous data

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values

        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """
        # retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        lbs = np.min(X, axis=0).tolist()
        ubs = np.max(X, axis=0).tolist()

        for i, (mn, mx) in enumerate(zip(lbs, ubs)):
            data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)

        return data

    def _predict(self, X: np.ndarray):
        response_values = self.get_response_values(X)

        # TODO: compute the predicted mean and variance of an RF given the response values of each trees

        means = response_values.mean(axis=1)
        vars_ = response_values.var(axis=1)

        return means.flatten(), vars_.flatten()

    def get_response_values(self, X: np.ndarray):
        """
        Collect the response values of each individual trees.
        """
        all_preds = []
        third_dimension = 0

        for row_X in X:
            preds_per_tree = self.rf.all_leaf_values(row_X)
            all_preds.append(preds_per_tree)
            max_num_leaf_data = max(map(len, preds_per_tree))
            third_dimension = max(max_num_leaf_data, third_dimension)
        # Transform list of 2d arrays into a 3d array
        response_values = np.zeros((X.shape[0], self.rf_opts.num_trees, third_dimension)) * np.NaN

        for i, preds_per_tree in enumerate(all_preds):
            for j, pred in enumerate(preds_per_tree):
                response_values[i, j, :len(pred)] = pred
        response_values = np.nanmean(response_values, axis=2)
        return response_values
