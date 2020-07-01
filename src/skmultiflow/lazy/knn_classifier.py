from skmultiflow.core import ClassifierMixin
from skmultiflow.lazy.base_neighbors import BaseNeighbors
from skmultiflow.utils.utils import *

import warnings


def KNN(n_neighbors=5, max_window_size=1000,
        leaf_size=30):     # pragma: no cover
    warnings.warn("'KNN' has been renamed to 'KNNClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)
    return KNNClassifier(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         )


class KNNClassifier(BaseNeighbors, ClassifierMixin):
    """ k-Nearest Neighbors classifier.

    This non-parametric classification method keeps track of the last
    ``max_window_size`` training samples. The predicted class-label for a
    given query sample is obtained in two steps:

    1. Find the closest n_neighbors to the query sample in the data window.
    2. Aggregate the class-labels of the n_neighbors to define the predicted
       class for the query sample.

    Parameters
    ----------
    n_neighbors: int (default=5)
        The number of nearest neighbors to search for.

    max_window_size: int (default=1000)
        The maximum size of the window storing the last observed samples.

    leaf_size: int (default=30)
        sklearn.KDTree parameter. The maximum number of samples that can
        be stored in one leaf node, which determines from which point the
        algorithm will switch for a brute-force approach. The bigger this
        number the faster the tree construction time, but the slower the
        query time will be.

    metric: string or sklearn.DistanceMetric object
        sklearn.KDTree parameter. The distance metric to use for the KDTree.
        Default=’euclidean’. KNNClassifier.valid_metrics() gives a list of
        the metrics which are valid for KDTree.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.lazy import KNNClassifier
    >>> from skmultiflow.data import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(random_state=1, noise_percentage=.1)
    >>> knn = KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40)
    >>> # Keep track of sample count and correct prediction count
    >>> n_samples = 0
    >>> corrects = 0
    >>> while n_samples < 5000:
    ...     X, y = stream.next_sample()
    ...     my_pred = knn.predict(X)
    ...     if y[0] == my_pred[0]:
    ...         corrects += 1
    ...     knn = knn.partial_fit(X, y)
    ...     n_samples += 1
    >>>
    >>> # Displaying results
    >>> print('KNNClassifier usage example')
    >>> print('{} samples analyzed.'.format(n_samples))
    5000 samples analyzed.
    >>> print("KNNClassifier's performance: {}".format(corrects/n_samples))
    KNN's performance: 0.8776

    """

    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean',
                 standardize=False
                 ):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric,
                         standardize=standardize
                         )
        self.classes = []
        self._mean = None
        self._square_different_to_mean = None
        self.number_of_instances = 0

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification targets for all
            samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known classes.

        sample_weight: Not used.

        Returns
        -------
        KNNClassifier
            self

        Notes
        -----
        For the K-Nearest Neighbors Classifier, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the size_limit is reached, removing older results. To store
        the viewed samples we use a InstanceWindow object. For this class'
        documentation please visit skmultiflow.core.utils.data_structures

        """
        r, c = get_dimensions(X)
        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

        if self.standardize:
            if  self._mean is None:
                self._mean = np.zeros((r, c))

            if  self._square_different_to_mean is None:
                self._square_different_to_mean = np.zeros((r, c))

            self.number_of_instances += 1
            self.update_statistics(X)

        for i in range(r):
            self.data_window.add_sample(X[i], y[i])

        return self

    def predict(self, X):
        """ Predict the class label for sample X

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.

        """
        if self.standardize:
            standardize_instance = self.apply_standardize(X)

        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        """ Estimate the probability of X belonging to each class-labels.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            A  2D array of shape (n_samples, n_classes). Where each i-th row
            contains len(self.target_value) elements, representing the
            probability that the i-th sample of X belongs to a certain
            class label.

        """
        self.number_of_instances += 1
        r, c = get_dimensions(X)
        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # The model is empty, defaulting to zero
            return np.zeros(shape=(r, 1))
        proba = []

        self.classes = list(set().union(self.classes,
                                        np.unique(self.data_window.targets_buffer.astype(np.int))))

        new_dist, new_ind = self._get_neighbors(X)
        for i in range(r):
            votes = [0.0 for _ in range(int(max(self.classes) + 1))]
            for index in new_ind[i]:
                votes[int(self.data_window.targets_buffer[index])] += 1. / len(new_ind[i])
            proba.append(votes)

        return np.asarray(proba)

    def update_statistics(self, X):
        """
        The Welford's method  is used to calculate the moving standard deviation.
        we named (n-1)S2 in equation 1 square different to mean
        equation1: (n-1)s^2 = sigma_1_to_n(x_i - mean(x)_n )^2
        """
        differential = np.subtract(X, self._mean)/self.number_of_instances
        new_mean = self._mean + differential
        new_square_different_to_mean_increment = \
            np.subtract(X, new_mean) * np.subtract(X, self._mean)
        self._mean = new_mean
        self._square_different_to_mean = \
            self._square_different_to_mean + new_square_different_to_mean_increment



    @property
    def get_mean(self):
        return self._mean

    @property
    def get_sd(self):
        return np.sqrt(self._square_different_to_mean / self.number_of_instances)

    def apply_standardize(self, X):
#        return np.subtract(X, self._mean)/self.get_sd
        r, c = get_dimensions(self.get_sd)
        if c != np.count_nonzero(self.get_sd):
                return X

        return np.divide(np.subtract(X, self._mean), self.get_sd)

