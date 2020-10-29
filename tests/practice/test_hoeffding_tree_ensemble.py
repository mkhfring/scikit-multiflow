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
from tests import TEST_DIRECTORY

from array import array


class AwsomeHoeffdingTree(HoeffdingTreeClassifier):
    def __init__(self, grace_period=200, split_confidence=0.0000001,
                 tie_threshold=0.05, classes=None):
        super().__init__(grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         classes=classes)
        self.evaluator_method = ClassificationPerformanceEvaluator
        self.evaluator = self.evaluator_method()


class HoeffdingTreeEnsemble(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=None, window_size=100, n_estimators=1, classes=None):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.base_estimator = AwsomeHoeffdingTree
        self.classes = classes
        self.ensemble = None
        self.new_classifier_trigger = -1
        self.X_batch = None
        self.y_batch = None
        self.ensemble_candidate = None
        self.number_of_correct_predictions = 0
        self.accuracy_per_sample = []

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        if self.ensemble is None:
            self._init_ensembles(X, y, self.classes)

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

        for i in range(len(self.ensemble)):
            vote = cp.deepcopy(self.ensemble[i].get_votes_for_instance(X))

            if vote != {} and sum(vote.values()) > 0:
                vote = normalize_values_in_dict(vote, inplace=True)
                if self.classes:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                    y_proba_dict = {index: value for index, value in enumerate(y_proba)}

                performance = self.ensemble[i].evaluator.accuracy_score()
                if performance != 0.0:
                    for k in vote:
                        # Multiplying the votes by the performance of each the hoeffding tees in the ensemble
                        vote[k] = vote[k] * performance

                for key, value in vote.items():
                    y_proba_dict[float(key)] = value

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
            classes=classes
        ).partial_fit(X, y, classes)
        self.ensemble = [self.base_estimator(
            grace_period=randint(10, 200),
            split_confidence=uniform(0, 1),
            tie_threshold=uniform(0, 1),
            classes=classes
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
            n_estimators=10,
            classes=classes
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

        return average_proba



def test_hoeffding_tree_ensemble():
    test_data_directory = os.path.join(TEST_DIRECTORY, 'data')
    test_file = os.path.join(
        test_data_directory,
        'test_data/electricity.csv'
    )
    raw_data = pd.read_csv(test_file)
    stream1 = DataStream(raw_data, name='Test')
    stream2 = RandomTreeGenerator(
        tree_random_state=23, sample_random_state=12, n_classes=4,
        n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
        max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15
    )
#    learner = HoeffdingTreeClassifier(
#        leaf_prediction='nb',
#        classes=stream.target_values
#    )

#    learner = HoeffdingTreeEnsemble(
#        n_estimators=3,
#        classes=stream.target_values)
#    learner.classes = stream.target_values
    learner = DeepStreamLearner(classes=stream1.target_values)
    stream1_learner = calculate_accuracy(learner, stream1, 100)
    learner = DeepStreamLearner(classes=stream2.target_values)
    stream2_learner = calculate_accuracy(learner, stream2, 100)
    with open (
        os.path.join(test_data_directory, 'test_data/test_result.txt'),
        '+w'
    ) as f:
        f.write('stream1 accuracy: {} \n'.format(stream1_learner.accuracy[-1]))
        f.write('stream1 first_layer_accuracy: {} \n'.format(stream1_learner.first_layer_cascade.accuracy_per_sample[-1]))
        f.write('stream1 average_accuracy: {} \n'.format(sum(stream1_learner.accuracy)/stream1_learner.number_of_samples))
        f.write('stream1 first_layer_average_accuracy: {} \n \n'.format(sum(stream1_learner.first_layer_cascade.accuracy_per_sample)/stream1_learner.number_of_samples))
        f.write('stream2 accuracy: {} \n'.format(stream2_learner.accuracy[-1]))
        f.write('stream2 first_layer_accuracy: {} \n'.format(stream2_learner.first_layer_cascade.accuracy_per_sample[-1]))
        f.write('stream2 average_accuracy: {} \n'.format(sum(stream2_learner.accuracy)/stream2_learner.number_of_samples))
        f.write('stream2 first_layer_average_accuracy: {} \n \n'.format(sum(stream2_learner.first_layer_cascade.accuracy_per_sample)/stream2_learner.number_of_samples))

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    assert 1 == 1



def calculate_accuracy(learner, stream, max_samples=0):
    cnt = 0
    max_samples = max_samples
    predictions = array('i')
    proba_predictions = []
    wait_samples = 1
    correct_predictions = 0

    while stream.has_more_samples() and cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):

            y_pred = learner.predict(X, y)
            if y_pred == y:
                correct_predictions +=1


#            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    return learner


#    cnt = 0
#    max_samples = 100000
#    predictions = array('i')
#    proba_predictions = []
#    wait_samples = 1
#    correct_predictions = 0
#
#    while stream.has_more_samples() and cnt < max_samples:
#        X, y = stream.next_sample()
#        # Test every n samples
#        if (cnt % wait_samples == 0) and (cnt != 0):
#
#            y_pred = learner.predict(X, y)
#            if y_pred == y:
#                correct_predictions +=1
#
#
##            proba_predictions.append(learner.predict_proba(X)[0])
#        learner.partial_fit(X, y)
#        cnt += 1

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    expected_info = "HoeffdingTreeClassifier(binary_split=False, grace_period=200, leaf_prediction='nb', " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], remove_poor_atts=False, " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info
#    deep_stream_learner = DeepStreamLearner()
#    metrics = ['accuracy']
#    output_file = os.path.join('data', 'test_data/ensemble_output.csv')
#    evaluator = EvaluatePrequential(
#        max_samples=stream.n_samples,
#        metrics=metrics,
#        pretrain_size=500,
#        output_file=output_file
#    )
#    hoeffding_tree_learner = HoeffdingTreeClassifier(
#    )
#    result = evaluator.evaluate(
#        stream=stream,
#        model=[
#            deep_stream_learner,
#            hoeffding_tree_learner
#
#        ]
#    )
#
#    mean_performance, current_performance = evaluator.get_measurements()
#    print(current_performance[0].accuracy_score())
#    import pudb; pudb.set_trace()  # XXX BREAKPOINT
#    assert 1 == 1
