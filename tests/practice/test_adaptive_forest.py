
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
from skmultiflow.data import RandomTreeGenerator, SEAGenerator
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, StreamingRandomPatchesClassifier
from tests import TEST_DIRECTORY


class ExtendedHoeffdingAdaptiveTree(HoeffdingAdaptiveTreeClassifier):
    def __init__(self, grace_period=200, split_confidence=0.0000001,
                 tie_threshold=0.05, classes=None):
        super().__init__(grace_period=grace_period, split_confidence=split_confidence, tie_threshold=tie_threshold)
        self.evaluator_method = ClassificationPerformanceEvaluator
        self.evaluator = self.evaluator_method()
        self.accuracy = 0


class AdaptiveHoeffdingTreeEnsemble(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=None, window_size=100, n_estimators=1, classes=None):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = ExtendedHoeffdingAdaptiveTree
        self.classes = classes
        self.ensemble = None
        self.new_classifier_trigger = -1
        self.X_batch = None
        self.y_batch = None
        self.ensemble_candidate = None
        self.number_of_correct_predictions = 0
        self.accuracy_per_sample = []
        self.estimators_votes = None

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        if self.ensemble is None:
            #            X_random_sample = self.random_sample(X)
            #            self._init_ensembles(X, y, self.classes)
            self._init_ensembles(X, y, self.classes)
            assert 1 == 1

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
                            tie_threshold=uniform(0, 1),
                            classes=self.classes
                        ).partial_fit(self.X_batch, self.y_batch)
                    else:
                        self.ensemble_candidate = self.base_estimator(
                            grace_period=randint(10, 200),
                            split_confidence=uniform(0, 1),
                            tie_threshold=uniform(0, 1),
                            classes=self.classes
                        ).partial_fit(self.X_batch, self.y_batch)

                    self.new_classifier_trigger = 0

        return self

    def random_sample(self, X):
        import random
        from math import sqrt, ceil

        rows, columns = X.shape
        sample = np.zeros((rows, ceil(sqrt(columns))))
        for row in range(rows):
            column_sample_index = random.sample(
                range(columns),
                ceil(sqrt(columns))
            )
            for index, column_index in enumerate(sorted(column_sample_index)):
                sample[row][index] = X[row][column_index]

        return sample


    def predict(self, X):
        y_proba = self.predict_proba(X)
        if len(y_proba) != 2:
            assert 1 == 1
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
                if self.classes:
                    y = [0 for i in range(len(self.classes))]
                    y_proba.append(y)
                else:
                    y_proba.append([0])

                #                    y_proba.append([i for i in range(max(self.classes) +1))])

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
        #        if np.shape(y_proba)[1] == 2:
        #            import pudb; pudb.set_trace()  # XXX BREAKPOINT
        #            assert 1 == 1
        return y_proba

    def get_votes_for_instance(self, X):
        combined_votes = {}
        self.estimators_votes = None

        for i in range(len(self.ensemble)):
            vote = cp.deepcopy(self.ensemble[i].get_votes_for_instance(X))
            if hasattr(self.ensemble[i], 'predict_proba'):
                ensemble_class_distribution = self.ensemble[i].predict_proba([X])

            if self.estimators_votes is None:
                self.estimators_votes = ensemble_class_distribution
            else:
                self.estimators_votes = np.concatenate(
                    (self.estimators_votes, ensemble_class_distribution),
                    axis=1
                )



            if vote != {} and sum(vote.values()) > 0:
                vote = normalize_values_in_dict(vote, inplace=True)
                y_proba_dict = None
                if self.classes:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                    y_proba_dict = {index: value for index, value in enumerate(y_proba)}

                performance = self.ensemble[i].evaluator.accuracy_score()
                if performance != 0.0:
                    for k in vote:
                        # Multiplying the votes by the performance of each the hoeffding tees in the ensemble
                        vote[k] = vote[k] * performance

                if y_proba_dict:
                    for key, value in vote.items():
                        y_proba_dict[float(key)] = value

                y_proba_dict = vote

                # Add values
                for k in vote:
                    try:
                        # Combining the result predicted by each classifier for each instance
                        #                        combined_votes[k] += vote[k]
                        combined_votes[k] += y_proba_dict[k]
                    except KeyError:
                        #                        combined_votes[k] = vote[k]
                        combined_votes[k] = y_proba_dict[k]
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
            tie_threshold=uniform(0, 1),
        ).partial_fit(X, y, classes)
        self.ensemble = [self.base_estimator(
            grace_period=randint(10, 200),
            split_confidence=uniform(0, 1),
            tie_threshold=uniform(0, 1),
        ).partial_fit(X, y, classes) for e in range(
            self.n_estimators)]

        return self

    def _test_and_train_ensembles_and_condidate(self, X, y, classes=None):
        #        X_sample = self.random_sample(X)
        candidate_prediction = self.ensemble_candidate.predict(X)
        for index, prediction in enumerate(candidate_prediction):
            self.ensemble_candidate.evaluator.add_result(prediction, y[index])

        #        self.ensemble_candidate.partial_fit(X, y, classes)
        self.ensemble_candidate.partial_fit(X, y, classes)
        for classifier in self.ensemble:
            y_pred = classifier.predict(X)
            for index, prediction in enumerate(y_pred):
                classifier.evaluator.add_result(prediction, y[index])

            classifier.partial_fit(X, y, classes)

        return self


class DeepStreamLearner(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, n_ensembels=2, classes=None):
        self.base_ensemble_learner = AdaptiveHoeffdingTreeEnsemble(
            n_estimators=4
        )
        self.n_ensembles = n_ensembels
        self.ensembel_learners = None
        self.last_layer_cascade = None
        self.number_of_samples = 0
        self.accuracy = []
        self.number_of_correct_predictions = 0

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
        self.number_of_samples += np.shape(X)[0]

        if self.ensembel_learners is None:
            self._init_cascades(X, y)

        else:
            self.first_layer_cascade.partial_fit(X, y)
            first_layer_prediction = self.first_layer_cascade.predict_proba(X)
            extended_features = np.concatenate((X, first_layer_prediction), axis=1)
            if np.shape(extended_features)[1] == 7:
                pass

            self.last_layer_cascade.partial_fit(extended_features, y)
            return self


    def predict(self, X, y=None):
#        return self.last_layer_cascade.predict(X)
        y_proba = self.predict_proba(X, y)
        n_rows = y_proba.shape[0]
        y_pred = np.zeros(n_rows, dtype=int)
        for i in range(n_rows):
            index = np.argmax(y_proba[i])
            y_pred[i] = index
        if y_pred == y:
            self.number_of_correct_predictions += 1

        self.accuracy.append(
            self.number_of_correct_predictions / self.number_of_samples
        )
        return y_pred

    def predict_proba(self, X, y=None):
        first_layer_predict_proba = self.first_layer_cascade.predict_proba(X)

        # Try to track the accuracy per samples for the first layer
        first_layer_prediction = self.first_layer_cascade.predict(X)
        if first_layer_prediction == y:
            self.first_layer_cascade.number_of_correct_predictions +=1
        self.first_layer_cascade.accuracy_per_sample.append(

            self.first_layer_cascade.number_of_correct_predictions / self.number_of_samples
        )

        extended_features = np.concatenate(
            (X, self.first_layer_cascade.estimators_votes),
#            (X, first_layer_predict_proba),
            axis=1
        )
        second_layer_predict_proba = self.last_layer_cascade.predict_proba(
            extended_features
        )
        second_layer_prediction = self.last_layer_cascade.predict(
            extended_features
        )
        if second_layer_prediction == y:
            self.last_layer_cascade.number_of_correct_predictions +=1

        self.last_layer_cascade.accuracy_per_sample.append(

            self.last_layer_cascade.number_of_correct_predictions / self.number_of_samples
        )

        average_proba = (
            first_layer_predict_proba + second_layer_predict_proba
        ) / 2
        return average_proba

#        return second_layer_predict_proba


def test_adaptive_forest():
    test_data_directory = os.path.join(TEST_DIRECTORY, 'data')
    test_file = os.path.join(
        test_data_directory,
        'test_data/weather.csv'
    )
    raw_data = pd.read_csv(test_file)
    stream1 = DataStream(raw_data, name='Test')
    stream2 = DataStream(raw_data, name='Test')
#    learner = ExtendedHoeffdingAdaptiveTree()
    learner1 = AdaptiveHoeffdingTreeEnsemble(n_estimators=4)
    # stream1_learner = calculate_accuracy(learner, stream1, stream1.n_samples)
#    stream2_learner = calculate_accuracy(learner1, stream2, stream2.n_samples)
#    stream1_learner = calculate_accuracy(learner, stream1, stream1.n_samples)
#    learner3 = AdaptiveRandomForestClassifier(n_estimators=3)
#    stream3_learner = calculate_accuracy(learner3, stream1, stream1.n_samples)
#    learner4 = StreamingRandomPatchesClassifier(n_estimators=3)
#    stream4_learner = calculate_accuracy(learner4, stream1, stream1.n_samples)
    learner5 = DeepStreamLearner(classes=stream1.target_values)
    stream5_learner = calculate_accuracy(learner5, stream1, stream1.n_samples)

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    assert 1 == 1
#    print(stream2_learner.base_estimator.accuracy)
    with open (
            os.path.join(test_data_directory, 'test_data/adaptive_test_result.txt'),
            '+w'
    ) as f:
        f.write('stream2 average_accuracy:')

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    assert 1 == 1


def calculate_accuracy(learner, stream, max_samples=0):
    cnt = 0
    max_samples = max_samples
    proba_predictions = []
    wait_samples = 1
    correct_predictions = 0

    while stream.has_more_samples() and cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):

            y_pred = learner.predict(X)
            if y_pred == y:
                correct_predictions +=1


            #            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1
#    if hasattr(learner, 'base_estimator'):
#        learner.base_estimator.accuracy = (correct_predictions / stream.n_samples)
#
#    else:
#        learner.accuracy = (correct_predictions / stream.n_samples)

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    return learner
