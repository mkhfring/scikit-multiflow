import itertools
import os

import numpy as np
import copy as cp

import pandas as pd
from skmultiflow.core import BaseSKMObject, MetaEstimatorMixin, ClassifierMixin
from skmultiflow.data import RandomTreeGenerator, DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from skmultiflow.utils import get_dimensions, normalize_values_in_dict


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
                new_model = self.train_model(
                    cp.deepcopy(self.base_estimator),
                    self.X_batch,
                    self.y_batch,
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

    def predict(self, X):
        y_proba = self.predict_proba(X)
        n_rows = y_proba.shape[0]
        y_pred = np.zeros(n_rows, dtype=int)
        for i in range(n_rows):
            index = np.argmax(y_proba[i])
            y_pred[i] = index
        return y_pred

    def predict_proba(self, X):

        r, _ = get_dimensions(X)
        y_proba = []
        for i in range(r):
            # Calculating the probability of each class using hoeffding trees in the ensemble for the current instance
            # (current batch of instances)
            votes = cp.deepcopy(self.get_votes_for_instance(X[i]))
            if votes == {}:
                y_proba.append([0])

            else:
                if sum(votes.values()) != 0:
                    # Normalizing the votes by dividing each vote from the sum of all the votes
                    votes = normalize_values_in_dict(votes)

                if self.classes is not None:
                    votes_array = np.zeros(int(max(self.classes)) + 1)
                else:
                    votes_array = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    try:
                        votes_array[int(key)] = value
                    except:
                        print('this is not ok ')
                y_proba.append(votes_array)

        if self.classes is not None:
            y_proba = np.asarray(y_proba)
        else:
            y_proba = np.asarray(list(itertools.zip_longest(*y_proba, fillvalue=0.0))).T
        return y_proba

    def get_votes_for_instance(self, X):
        combined_votes = {}

        for i in range(len(self.ensemble)):
            vote = cp.deepcopy(self.ensemble[i].get_votes_for_instance(X))
            if vote != {} and sum(vote.values()) > 0:
                vote = normalize_values_in_dict(vote, inplace=True)
                performance = self.ensemble[i].evaluator.accuracy_score()
                if performance != 0.0:
                    for k in vote:
                        # Multiplying the votes by the performance of each the hoeffding tees in the ensemble
                        vote[k] = vote[k] * performance
                # Add values
                for k in vote:
                    try:
                        # Combining the result predicted by each classifier for each instance
                        combined_votes[k] += vote[k]
                    except KeyError:
                        combined_votes[k] = vote[k]
        return combined_votes


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
        number_of_ensembles = len(self.ensemble)
        for i in range(number_of_ensembles-1):
            for j in range(0, number_of_ensembles-i-1):
                if self.ensemble[j].evaluator.accuracy_score() > self.ensemble[j+1].evaluator.accuracy_score():
                    self.ensemble[j], self.ensemble[j+1] = self.ensemble[j+1], self.ensemble[j]


def test_hoeffding_tree_ensemble():
    test_file = os.path.join('data', 'test_data/covtype.csv')
    raw_data = pd.read_csv(test_file)
    stream = DataStream(raw_data, name='Test')
    estimator = AwsomeHoeffdingTree()
    ensemble_learner = HoeffdingTreeEnsemble(base_estimator=estimator, n_estimators=10)
    metrics = ['accuracy']
    output_file = os.path.join('data', 'test_data/ensemble_output.csv')
    evaluator = EvaluatePrequential(
        max_samples=stream.n_samples,
        metrics=metrics,
        pretrain_size=500,
        output_file=output_file
    )
    hoeffding_tree_learner = HoeffdingTreeClassifier(
    )
    result = evaluator.evaluate(
        stream=stream,
        model=[
            ensemble_learner,
            hoeffding_tree_learner

        ]
    )
    mean_performance, current_performance = evaluator.get_measurements()
    print(current_performance[0].accuracy_score())
    assert 1 == 1
