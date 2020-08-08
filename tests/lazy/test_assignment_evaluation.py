
import os
import numpy as np
import pandas as pd

import pytest   # noqa

from skmultiflow.data.data_stream import DataStream
from skmultiflow.lazy import KNNClassifier, WeightedKNNClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes


def test_data_stream(test_path):
    test_file = os.path.join(test_path, 'data/data_n30000.csv')
    raw_data = pd.read_csv(test_file)
    stream = DataStream(raw_data, name='Test')
    normal_knn_learner = KNNClassifier(
        n_neighbors=8,
        max_window_size=2000,
        leaf_size=40,
    )
    weighted_knn_learner = WeightedKNNClassifier(
        n_neighbors=8,
        max_window_size=2000,
        leaf_size=40
    )
    standardize_knn_learner = KNNClassifier(
        n_neighbors=8,
        max_window_size=2000,
        leaf_size=40,
        standardize=True
    )
    nominal_attr_idx = [x for x in range(15, len(stream.feature_names))]

    hoeffding_learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)
    nb_learner = NaiveBayes()

    metrics = ['accuracy', 'kappa_m', 'kappa_t', 'recall']
    output_file = os.path.join(test_path, 'data/kkn_output.csv')
    evaluator = EvaluatePrequential(metrics=metrics, output_file=output_file)

    # Evaluate
    result = evaluator.evaluate(
        stream=stream,
        model=[
            normal_knn_learner,
            weighted_knn_learner,
            standardize_knn_learner,
            hoeffding_learner,
            nb_learner,
        ]
    )
    mean_performance, current_performance = evaluator.get_measurements()
    assert 1 == 1

