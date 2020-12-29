from utils import *
import numpy as np
import itertools


class PredictionRunData:
    def __init__(self, color_model, similarity_model):
        self._minimize = 'mse' in similarity_model
        self._train_pred_raw, self._train_true_raw, self.train_meta = read_predictions_from_file(
            color_model_name=color_model, sample_set_name='train', metric_name=similarity_model)
        self._test_pred_raw, self._test_true_raw, self.test_meta = read_predictions_from_file(
            color_model_name=color_model, sample_set_name='test', metric_name=similarity_model)
        self._reject_pred_raw, self._reject_true_raw, self.reject_meta = read_predictions_from_file(
            color_model_name=color_model, sample_set_name='reject', metric_name=similarity_model)

    def mean_predictions_per_descriptor(self):
        return itertools.chain(self._train_pred_raw)

    def best_mean_predictions_per_class(self):

        def minimas_from_prediction(predictions_batch):
            def sort(scores):
                return sorted(scores.items(), key=lambda x: x[1], reverse=False)

            return [sort(combine_mean_scores(p))[0]
                    for predictions in predictions_batch for p in predictions]

        def maximas_from_predictions(predictions_batch):
            def sort(scores):
                return sorted(scores.items(), key=lambda x: x[1], reverse=True)

            return [sort(combine_mean_scores(p))[0]
                    for predictions in predictions_batch for p in predictions]

        def combine_mean_scores(predictions):
            return {cls: np.nanmean([np.nanmean(s) for s in scores])
                    for cls, scores in predictions.items()}

        if self._minimize:
            return minimas_from_prediction(self.train_pred()), \
                   minimas_from_prediction(self.test_pred()), \
                   minimas_from_prediction(self.reject_pred())
        else:
            return maximas_from_predictions(self.train_pred()), \
                   maximas_from_predictions(self.test_pred()), \
                   maximas_from_predictions(self.reject_pred())

    """
    The following methods combine predictions of all training batches into one list.
    The returned value is a list of per-descriptor similarities of a sample to all trained descriptors.
    """
    def train_pred(self):
        return list(itertools.chain.from_iterable(self._train_pred_raw))

    def test_pred(self):
        return list(itertools.chain.from_iterable(self._test_pred_raw))

    def reject_pred(self):
        return list(itertools.chain.from_iterable(self._reject_pred_raw))

    """
    Combines true values of all training batches into one list.
    The returned value is a list of true classes corresponding to predictions.
    """
    def train_true(self):
        return list(itertools.chain.from_iterable(self._train_true_raw))

    def test_true(self):
        return list(itertools.chain.from_iterable(self._test_true_raw))

    def reject_true(self):
        return list(itertools.chain.from_iterable(self._reject_true_raw))
