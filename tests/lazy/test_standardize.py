

from array import array
from skmultiflow.lazy import WeightedKNNClassifier
from skmultiflow.data import SEAGenerator
import numpy as np


def test_standardize():
    stream = SEAGenerator(random_state=1)

    learner = WeightedKNNClassifier(
        n_neighbors=8,
        max_window_size=2000,
        leaf_size=40,
        standardize=True
    )

    # Test for Implementing moving average
    example_features = np.array(
        [
            [1, 2],
            [2, 4],
            [3, 9]
        ]
    )
    example_targets = np.array(
        [
            [1],
            [1],
            [1]
        ]
    )
    for i in range(len(example_features)):
        learner.partial_fit(
            example_features[i],
            example_targets[i]
        )

    moving_average = learner.get_mean
    assert type(moving_average) is np.ndarray
    assert np.alltrue(moving_average == np.mean(example_features, axis=0))
    moving_sd = learner.get_sd
    assert type(moving_sd) is np.ndarray
    assert np.alltrue(moving_sd == np.std(example_features, axis=0))
#
#    for value in example_array:
#        learner.update_deviation(value)
#
#    moving_sd = learner.get_deviation

#    cnt = 0
#    max_samples = 5000
#    predictions = array('i')
#    correct_predictions = 0
#    wait_samples = 100
#    X_batch = []
#    y_batch = []
#
#    while cnt < max_samples:
#        X, y = stream.next_sample()
#        X_batch.append(X[0])
#        y_batch.append(y[0])
#        # Test every n samples
#        if (cnt % wait_samples == 0) and (cnt != 0):
#            predictions.append(learner.predict(X)[0])
#            if y[0] == predictions[-1]:
#                correct_predictions += 1
#        learner.partial_fit(X, y)
#        cnt += 1
#
#    assert 1 == 1
#
#    expected_predictions = array('i', [1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
#                                       1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
#                                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#                                       0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
#                                       1, 1, 0, 1, 0, 0, 1, 0, 1])
#    assert np.alltrue(predictions == expected_predictions)
#
#    expected_correct_predictions = 49
#    assert correct_predictions == expected_correct_predictions
#
#    expected_info = "WeightedKNNClassifier(leaf_size=40, max_window_size=2000, " \
#                    "metric='euclidean', n_neighbors=8)"
#    info = " ".join([line.strip() for line in learner.get_info().split()])
#    assert info == expected_info
#
#    learner.reset()
#    info = " ".join([line.strip() for line in learner.get_info().split()])
#    assert info == expected_info
#
#    X_batch = np.array(X_batch)
#    y_batch = np.array(y_batch)
#    learner.fit(X_batch[:4500], y_batch[:4500], classes=[0, 1])
#    predictions = learner.predict(X_batch[4501:4550])
#
#    expected_predictions = array('i', [1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
#                                       1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
#                                       1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
#                                       0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
#                                       1, 1, 1, 1, 1, 1, 0, 1, 1])
#    assert np.alltrue(predictions == expected_predictions)
#
#    correct_predictions = sum(predictions == y_batch[4501:4550])
#    expected_correct_predictions = 48
#    assert correct_predictions == expected_correct_predictions
#
#    assert type(learner.predict(X)) == np.ndarray
#    assert type(learner.predict_proba(X)) == np.ndarray
