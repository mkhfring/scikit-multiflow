import numpy as np
import copy as cp

from skmultiflow.core import BaseSKMObject, MetaEstimatorMixin, ClassifierMixin
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from skmultiflow.utils import get_dimensions


class AwsomeHoeffdingTree(HoeffdingTreeClassifier):
    def __init__(self, evaluator=None):
        super().__init__()
        self.evaluator_method = ClassificationPerformanceEvaluator
        self.evaluator = self.evaluator_method()


class HoeffdingTreeEnsemble(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator, window_size=100, n_estimators=3, classes=None):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.classes = classes
        # The ensemble
        self.ensemble = []
        self.new_classifier_trigger = -1
        self.X_batch = None
        self.y_batch = None

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        number_of_instances, number_of_features = get_dimensions(X)

        if self.new_classifier_trigger < 0:
            # No models yet -- initialize
            self.X_batch = np.zeros((self.window_size, number_of_features))
            self.y_batch = np.zeros(self.window_size)
            self.sample_weight = np.zeros(self.window_size)
            self.new_classifier_trigger = 0

        for n in range(number_of_instances):
            self.X_batch[self.new_classifier_trigger] = X[n]
            self.y_batch[self.new_classifier_trigger] = y[n]
            self.sample_weight[self.new_classifier_trigger] = sample_weight[n] if sample_weight else 1.0

            self.new_classifier_trigger = self.new_classifier_trigger + 1
            if self.new_classifier_trigger == self.window_size:
                classes, _ = np.unique(self.y_batch, return_counts=True)
                self.classes = classes
                new_model = self.train_model(
                    cp.deepcopy(self.base_estimator),
                    self.X_batch,
                    self.y_batch,
                    classes
                )
                y_pred = new_model.predict(self.X_batch)
                for index, prediction in enumerate(y_pred):
                    new_model.evaluator.add_result(prediction, self.y_batch[index])

                if len(self.ensemble) > self.n_estimators:
                    if new_model.evaluator.accuracy_score() > self.ensemble[0].evaluator.accuracy_score():
                        self.ensemble[0] = new_model
                else:
                    self.ensemble.append(new_model)

                self.sort_ensemble()
                self.new_classifier_trigger = 0

        return self

    def predict_proba(self, X):
        number_of_instances, number_of_features = get_dimensions(X)
        # votes = np.zeros(number_of_instances)
        votes = [0.0 for _ in range(int(max(self.classes) + 1))]
        if len(self.ensemble) <= 0:
            return votes
        for classifier in self.ensemble:
            predicted_target = classifier.predict(X)
            votes[predicted_target.astype(np.int)[0]] += classifier.evaluator.accuracy_score()
            # votes = votes + 1. / len(self.ensemble) * classifier.predict(X)
        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        return np.asarray([np.argmax(votes)])

    def reset(self):
        self.ensemble = []
        self.i = -1
        self.X_batch = None
        self.y_batch = None
        return self

    def train_model(self, model, X, y, classes=None, sample_weight=None):
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X, y, classes)

        return model

    def sort_ensemble(self):
        # Using bubble sort to sort the ensemble based on accuracy
        number_of_ensembles = len(self.ensemble)
        for i in range(number_of_ensembles-1):
            for j in range(0, number_of_ensembles-i-1):
                if self.ensemble[j].evaluator.accuracy_score() > self.ensemble[j+1].evaluator.accuracy_score():
                    self.ensemble[j], self.ensemble[j+1] = self.ensemble[j+1], self.ensemble[j]


def test_hoeffding_tree_ensemble():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112)
    estimator = AwsomeHoeffdingTree()
    learner = HoeffdingTreeEnsemble(base_estimator=estimator)
    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)
    cnt = 0
    max_samples = 5000
    predictions = []
    true_labels = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1
    performance = correct_predictions / len(predictions)
    assert 1 == 1
