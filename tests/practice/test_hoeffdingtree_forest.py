

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
from skmultiflow.meta import AdaptiveRandomForestClassifier, \
    StreamingRandomPatchesClassifier
from skmultiflow.meta.adaptive_random_forests import ARFBaseLearner
from tests import TEST_DIRECTORY


class ExtendedHoeffdingAdaptiveTreeClassifier(HoeffdingAdaptiveTreeClassifier):
    def __init__(self, max_byte_size, memory_estimate_period, grace_period,
                 split_criterion, split_confidence, tie_threshold,
                 binary_split, stop_mem_management,
                 no_preprune, leaf_prediction, nb_threshold,
                 nominal_attributes
                 ):

        super().__init__(max_byte_size, memory_estimate_period, grace_period,
                 split_criterion, split_confidence, tie_threshold,
                 binary_split, stop_mem_management,
                 no_preprune, leaf_prediction, nb_threshold,
                 nominal_attributes)

    def new_instance(self):
        return ExtendedHoeffdingAdaptiveTreeClassifier(max_byte_size=self.max_byte_size,
                                          memory_estimate_period=self.memory_estimate_period,
                                          grace_period=self.grace_period,
                                          split_criterion=self.split_criterion,
                                          split_confidence=self.split_confidence,
                                          tie_threshold=self.tie_threshold,
                                          binary_split=self.binary_split,
                                          stop_mem_management=self.stop_mem_management,
                                          no_preprune=self.no_preprune,
                                          leaf_prediction=self.leaf_prediction,
                                          nb_threshold=self.nb_threshold,
                                          nominal_attributes=self.nominal_attributes)

class HoeffdingForestClassifier(AdaptiveRandomForestClassifier):

    def __init__(self, n_estimators, classes):
        super().__init__(n_estimators, classes)
        self.accuracy_per_sample = []
        self.number_of_correct_predictions = 0

    def _init_ensemble(self, X):
        self._set_max_features(get_dimensions(X)[1])

        self.ensemble = [ARFBaseLearner(index_original=i,
                                        classifier=ExtendedHoeffdingAdaptiveTreeClassifier(
                                            max_byte_size=self.max_byte_size,
                                            memory_estimate_period=self.memory_estimate_period,
                                            grace_period=self.grace_period,
                                            split_criterion=self.split_criterion,
                                            split_confidence=self.split_confidence,
                                            tie_threshold=self.tie_threshold,
                                            binary_split=self.binary_split,
                                            stop_mem_management=self.stop_mem_management,
                                            no_preprune=self.no_preprune,
                                            leaf_prediction=self.leaf_prediction,
                                            nb_threshold=self.nb_threshold,
                                            nominal_attributes=self.nominal_attributes),
                                        instances_seen=self.instances_seen,
                                        drift_detection_method=self.drift_detection_method,
                                        warning_detection_method=self.warning_detection_method,
                                        is_background_learner=False)
                         for i in range(self.n_estimators)]


class DeepStreamLearner(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, n_ensembel_estimators=10, n_ensembels=2, classes=None):
        self.base_ensemble_learner = HoeffdingForestClassifier(
            n_ensembel_estimators,
            classes
        )
        self.n_ensembles = n_ensembels
        self.ensembel_learners = None
        self.last_layer_cascade = None
        self.number_of_samples = 0
        self.accuracy = []
        self.number_of_correct_predictions = 0
        self.classes = classes

    def _init_cascades(self, X, y):
        first_cascade = cp.deepcopy(self.base_ensemble_learner)
        first_cascade.partial_fit(X, y, self.classes)
        first_layer_class_distribution = first_cascade.predict_proba(X)

        extended_features = np.concatenate(
            (X, first_layer_class_distribution),
            axis=1
        )
        second_cascade = cp.deepcopy(self.base_ensemble_learner)
        second_cascade.partial_fit(extended_features, y, self.classes)
        self.ensembel_learners = [first_cascade, second_cascade]
        self.first_layer_cascade = first_cascade
        self.last_layer_cascade = second_cascade



    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        self.number_of_samples += np.shape(X)[0]

        if self.ensembel_learners is None:
            self._init_cascades(X, y)

        else:
            self.first_layer_cascade.partial_fit(X, y, self.classes)
            first_layer_prediction = self.first_layer_cascade.predict_proba(X)
            extended_features = np.concatenate((X, first_layer_prediction), axis=1)

            self.last_layer_cascade.partial_fit(extended_features, y, self.classes)
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
#            (X, self.first_layer_cascade.estimators_votes),
            (X, first_layer_predict_proba),
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

#        return average_proba

        return second_layer_predict_proba


def test_adaptive_forest():
    test_data_directory = os.path.join(TEST_DIRECTORY, 'data')
    test_file = os.path.join(
        test_data_directory,
        'test_data/weather.csv'
    )
    raw_data = pd.read_csv(test_file)
    stream1 = DataStream(raw_data, name='Test')
    stream2 = DataStream(raw_data, name='Test')
#    learner = HoeffdingAdaptiveTreeClassifier()
#    learner1 = AdaptiveHoeffdingTreeEnsemble(n_estimators=4)
#    stream1_learner = calculate_accuracy(learner, stream1, stream1.n_samples)
#    stream2_learner = calculate_accuracy(learner1, stream2, stream2.n_samples)
#    stream1_learner = calculate_accuracy(learner, stream1, stream1.n_samples)
#    learner3 = HoeffdingForestClassifier(n_estimators=3)
#    stream3_learner = calculate_accuracy(learner3, stream1, stream1.n_samples)
#    learner4 = StreamingRandomPatchesClassifier(n_estimators=3)
#    stream4_learner = calculate_accuracy(learner4, stream1, stream1.n_samples)
    learner5 = DeepStreamLearner(
        n_ensembel_estimators=3,
        classes=stream1.target_values
    )
    stream5_learner = calculate_accuracy(learner5, stream1, stream1.n_samples)

    assert 1 == 1
#    print(stream2_learner.base_estimator.accuracy)
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    with open (
            os.path.join(test_data_directory, 'test_data/adaptive_test_result.txt'),
            '+w'
    ) as f:
        f.write('stream2 average_accuracy:')

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
            pass

            y_pred = learner.predict(X, y)
            if y_pred == y:
                correct_predictions +=1


            #            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y, classes=stream.target_values)
        cnt += 1
#    if hasattr(learner, 'base_estimator'):
#        learner.base_estimator.accuracy = (correct_predictions / stream.n_samples)
#
#    else:
#        learner.accuracy = (correct_predictions / stream.n_samples)

    return learner
