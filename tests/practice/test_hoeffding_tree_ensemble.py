import itertools
import os
from random import randint, uniform

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
    def __init__(self, grace_period=200, split_confidence=0.0000001, tie_threshold=0.05):
        super().__init__(grace_period=grace_period, split_confidence=split_confidence, tie_threshold=tie_threshold)
        self.evaluator_method = ClassificationPerformanceEvaluator
        self.evaluator = self.evaluator_method()


class HoeffdingTreeEnsemble(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator, window_size=100, n_estimators=1, classes=None):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = AwsomeHoeffdingTree
        self.classes = classes
        self.ensemble = None
        self.new_classifier_trigger = -1
        self.X_batch = None
        self.y_batch = None
        self.ensemble_candidate = None

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        if self.ensemble is None:
            self._init_ensembles(X, y)

        else:
            number_of_instances, number_of_features = get_dimensions(X)
            if self.new_classifier_trigger < 0:
                # No models yet -- initialize
                self.X_batch = np.zeros((self.window_size, number_of_features))
                self.y_batch = np.zeros(self.window_size)
                self.sample_weight = np.zeros(self.window_size)
                self.new_classifier_trigger = 0

            for n in range(number_of_instances):
                if np.shape(X)[1] != np.shape(self.X_batch)[1]:
                    continue

                self.X_batch[self.new_classifier_trigger] = X[n]
                self.y_batch[self.new_classifier_trigger] = y[n]
                self.sample_weight[self.new_classifier_trigger] = sample_weight[n] if sample_weight else 1.0

                self.new_classifier_trigger = self.new_classifier_trigger + 1
                if self.new_classifier_trigger == self.window_size:
                    self._test_and_train_ensembles_and_condidate(self.X_batch, self.y_batch)
                    weakest_classifier_index = self.get_weakest_ensemble_classifier()
                    if self.ensemble_candidate.evaluator.accuracy_score() > \
                            self.ensemble[weakest_classifier_index].evaluator.accuracy_score():
                        self.ensemble[weakest_classifier_index] = self.ensemble_candidate
                        self.ensemble_candidate = self.base_estimator(
                            grace_period=randint(10, 200),
                            split_confidence=uniform(0, 1),
                            tie_threshold=uniform(0, 1)
                        ).partial_fit(self.X_batch, self.y_batch)
                    else:
                        self.ensemble_candidate = self.base_estimator(
                            grace_period=randint(10, 200),
                            split_confidence=uniform(0, 1),
                            tie_threshold=uniform(0, 1)
                        ).partial_fit(self.X_batch, self.y_batch)

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

    def get_weakest_ensemble_classifier(self):
        number_of_ensembles = len(self.ensemble)
        weakest_classifier_index = 0
        for i in range(number_of_ensembles-1):
            for j in range(0, number_of_ensembles-i-1):
                if self.ensemble[j].evaluator.accuracy_score() < self.ensemble[j+1].evaluator.accuracy_score():
                    weakest_classifier_index = j

        return weakest_classifier_index

    def _init_ensembles(self, X, y, classes=None):
        self.ensemble_candidate = self.base_estimator(
            grace_period=randint(10, 200),
            split_confidence=uniform(0, 1),
            tie_threshold=uniform(0, 1)
        ).partial_fit(X, y, classes)
        self.ensemble = [self.base_estimator(
            grace_period=randint(10, 200),
            split_confidence=uniform(0, 1),
            tie_threshold=uniform(0, 1)
        ).partial_fit(X, y, classes) for e in range(self.n_estimators)]
        return self

    def _test_and_train_ensembles_and_condidate(self, X, y, classes=None):
        candidate_prediction = self.ensemble_candidate.predict(X)
        for index, prediction in enumerate(candidate_prediction):
            self.ensemble_candidate.evaluator.add_result(prediction, y[index])

        self.ensemble_candidate.partial_fit(X, y, classes)
        for classifier in self.ensemble:
            y_pred = classifier.predict(X)
            for index, prediction in enumerate(y_pred):
                classifier.evaluator.add_result(prediction, y[index])

            classifier.partial_fit(X, y, classes)

        return self

class DeepStreamLearner(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, n_ensembels=2, classes=None):
        self.base_ensemble_learner = HoeffdingTreeEnsemble(
            base_estimator=AwsomeHoeffdingTree,
            n_estimators=3
        )
        self.n_ensembles = n_ensembels
        self.ensembel_learners = None
        self.last_layer_cascade = None

    def _init_cascades(self, X, y):
        first_cascade = cp.deepcopy(self.base_ensemble_learner)
        first_cascade.partial_fit(X, y)
        first_layer_prediction = first_cascade.predict_proba(X)
        extended_features = np.concatenate((X, first_layer_prediction), axis=1)
        second_cascade = cp.deepcopy(self.base_ensemble_learner)
        second_cascade.partial_fit(extended_features, y)
        self.ensembel_learners = [first_cascade, second_cascade]
        self.first_layer_cascade = first_cascade
        self.last_layer_cascade = second_cascade



    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        if self.ensembel_learners is None:
            self._init_cascades(X, y)

        else:
            pass
#            self.first_layer_cascade.partial_fit(X, y)
#            first_layer_prediction = self.first_layer_cascade.predict_proba(X)
#            extended_features = np.concatenate((X, first_layer_prediction), axis=1)
#            if np.shape(extended_features)[1] == 7:
#                pass
#
#            self.last_layer_cascade.partial_fit(extended_features, y)
#            return self

#            number_of_instances, number_of_features = get_dimensions(X)
#            if self.new_classifier_trigger < 0:
#                # No models yet -- initialize
#                self.X_batch = np.zeros((self.window_size, number_of_features))
#                self.y_batch = np.zeros(self.window_size)
#                self.sample_weight = np.zeros(self.window_size)
#                self.new_classifier_trigger = 0
#
#            for n in range(number_of_instances):
#                self.X_batch[self.new_classifier_trigger] = X[n]
#                self.y_batch[self.new_classifier_trigger] = y[n]
#                self.sample_weight[self.new_classifier_trigger] = sample_weight[n] if sample_weight else 1.0
#
#                self.new_classifier_trigger = self.new_classifier_trigger + 1
#                if self.new_classifier_trigger == self.window_size:
#                    self._test_and_train_ensembles_and_condidate(self.X_batch, self.y_batch)
#                    weakest_classifier_index = self.get_weakest_ensemble_classifier()
#                    if self.ensemble_candidate.evaluator.accuracy_score() > \
#                            self.ensemble[weakest_classifier_index].evaluator.accuracy_score():
#                        self.ensemble[weakest_classifier_index] = self.ensemble_candidate
#                        self.ensemble_candidate = self.base_estimator(
#                            grace_period=randint(10, 200),
#                            split_confidence=uniform(0, 1),
#                            tie_threshold=uniform(0, 1)
#                        ).partial_fit(self.X_batch, self.y_batch)
#                    else:
#                        self.ensemble_candidate = self.base_estimator(
#                            grace_period=randint(10, 200),
#                            split_confidence=uniform(0, 1),
#                            tie_threshold=uniform(0, 1)
#                        ).partial_fit(self.X_batch, self.y_batch)
#
#                    self.new_classifier_trigger = 0
#
#        return self

    def predict(self, X):
        return self.last_layer_cascade.predict(X)
#        y_proba = self.predict_proba(X)
#        n_rows = y_proba.shape[0]
#        y_pred = np.zeros(n_rows, dtype=int)
#        for i in range(n_rows):
#            index = np.argmax(y_proba[i])
#            y_pred[i] = index
#        return y_pred

    def predict_proba(self, X):
        return self.last_layer_cascade.predict_proba(X)

#        r, _ = get_dimensions(X)
#        y_proba = []
#        for i in range(r):
#            # Calculating the probability of each class using hoeffding trees in the ensemble for the current instance
#            # (current batch of instances)
#            votes = cp.deepcopy(self.get_votes_for_instance(X[i]))
#            if votes == {}:
#                y_proba.append([0])
#
#            else:
#                if sum(votes.values()) != 0:
#                    # Normalizing the votes by dividing each vote from the sum of all the votes
#                    votes = normalize_values_in_dict(votes)
#
#                if self.classes is not None:
#                    votes_array = np.zeros(int(max(self.classes)) + 1)
#                else:
#                    votes_array = np.zeros(int(max(votes.keys())) + 1)
#                for key, value in votes.items():
#                    try:
#                        votes_array[int(key)] = value
#                    except:
#                        print('this is not ok ')
#                y_proba.append(votes_array)
#
#        if self.classes is not None:
#            y_proba = np.asarray(y_proba)
#        else:
#            y_proba = np.asarray(list(itertools.zip_longest(*y_proba, fillvalue=0.0))).T
#        return y_proba



def test_hoeffding_tree_ensemble():
    test_file = os.path.join('data', 'test_data/electricity.csv')
    raw_data = pd.read_csv(test_file)
    stream = DataStream(raw_data, name='Test')
    deep_stream_learner = DeepStreamLearner()
    metrics = ['accuracy']
    output_file = os.path.join('data', 'test_data/ensemble_output.csv')
    evaluator = EvaluatePrequential(
        max_samples=stream.n_samples,
        metrics=metrics,
        pretrain_size=4000,
        output_file=output_file
    )
#    hoeffding_tree_learner = HoeffdingTreeClassifier(
#    )
    result = evaluator.evaluate(
        stream=stream,
        model=[
            deep_stream_learner

        ]
    )

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    mean_performance, current_performance = evaluator.get_measurements()
    print(current_performance[0].accuracy_score())
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    assert 1 == 1
