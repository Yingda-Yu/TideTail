import numpy as np
import pytest

from tidetail.metrics import event_peak_error, mae, rmse, tail_rmse


def test_hand_computed_overall_errors():
    assert mae([1, 2, 3], [2, 0, 3]) == pytest.approx(1)
    assert rmse([1, 2, 3], [2, 0, 3]) == pytest.approx(np.sqrt(5 / 3))
    assert rmse([2], [2]) == 0


def test_tail_cutoff_fitted_on_train_includes_ties():
    assert tail_rmse([0, 5, 10], [20, 4, 7], q=.5, train_values=[0, 5, 10]) == np.sqrt(5)
    assert tail_rmse([0, 5, 10], [20, 4, 7], threshold=5) == np.sqrt(5)
    assert tail_rmse([0, 5, 10], [20, 4, 7], q=1, train_values=[0, 5, 10]) == 3


def test_no_tail_returns_nan_and_train_cutoff_does_not_follow_test():
    assert np.isnan(tail_rmse([0, 1], [0, 2], train_values=[10, 20]))
    assert tail_rmse([100, 101], [99, 100], train_values=[10, 20]) == 1


def test_tail_requires_explicit_training_reference():
    with pytest.raises(ValueError, match="training threshold"):
        tail_rmse([0, 1], [0, 2])
    with pytest.raises(ValueError):
        tail_rmse([0, 1], [0, 2], train_values=[0, 1], threshold=1)


@pytest.mark.parametrize("q", [-.1, 1.1, np.nan, np.inf])
def test_bad_quantile(q):
    with pytest.raises(ValueError):
        tail_rmse([1], [1], q=q, train_values=[1])


def test_peaks_weight_events_equally_and_use_true_intervals():
    # Two true events: [2, 5] and [3]. Predicted outside-event peak 100 is ignored.
    assert event_peak_error([0, 2, 5, 0, 3], [100, 3, 3, 0, 6], threshold=2) == 2.5
    assert event_peak_error([2, 3, 2], [2, 3, 2], threshold=2) == 0
    assert event_peak_error([2], [5], threshold=2) == 3
    assert np.isnan(event_peak_error([0, 1], [0, 9], threshold=2))


@pytest.mark.parametrize("metric", [mae, rmse,
                                     lambda a, b: tail_rmse(a, b, threshold=1),
                                     lambda a, b: event_peak_error(a, b, threshold=1)])
@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([[1]], [[1]]),
                                  ([np.nan], [1]), ([1], [np.inf])])
def test_reject_invalid_metric_arrays(metric, a, b):
    with pytest.raises(ValueError):
        metric(a, b)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, [1, 2]])
def test_reject_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        tail_rmse([1], [1], threshold=threshold)
    with pytest.raises(ValueError):
        event_peak_error([1], [1], threshold=threshold)


@pytest.mark.parametrize("training", [[], [np.nan], [[1]]])
def test_reject_invalid_training_reference(training):
    with pytest.raises(ValueError):
        tail_rmse([1], [1], train_values=training)
