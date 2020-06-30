from .knn_classifier import KNNClassifier
from skmultiflow.utils.utils import *


class WeightedKNNClassifier(KNNClassifier):

    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean',
                 ):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric,
                         )
        self.classes = []

    def predict_proba(self, X):
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
            for enumerate_index, target_index in enumerate(new_ind[i]):
                votes[int(self.data_window.targets_buffer[target_index])] += 1. / new_dist[i][enumerate_index]
            proba.append(votes)

        return np.asarray(proba)
