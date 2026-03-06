"""Tests for the evaluation metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    interval_coverage,
    interval_width,
    mean_absolute_error,
    mean_absolute_percentage_error,
    pinball_loss,
    root_mean_squared_error,
    symmetric_mape,
    evaluate_forecaster,
)
from src.models.forecasting import ElectricityPriceForecaster
from tests.conftest import make_price_series


class TestPointMetrics:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(1.0)

    def test_mape_excludes_zeros(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        # Only the second element should be counted
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert result == pytest.approx(10.0)

    def test_smape_symmetric(self):
        y_true = np.array([100.0])
        y_pred = np.array([120.0])
        r1 = symmetric_mape(y_true, y_pred)
        r2 = symmetric_mape(y_pred, y_true)
        assert r1 == pytest.approx(r2, rel=1e-3)


class TestProbabilisticMetrics:
    def test_pinball_loss_at_median(self):
        # For q=0.5, pinball = 0.5 * MAE
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        expected = 0.5 * mean_absolute_error(y_true, y_pred)
        assert pinball_loss(y_true, y_pred, quantile=0.5) == pytest.approx(expected)

    def test_pinball_loss_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(y, y, quantile=0.5) == pytest.approx(0.0)
        assert pinball_loss(y, y, quantile=0.1) == pytest.approx(0.0)
        assert pinball_loss(y, y, quantile=0.9) == pytest.approx(0.0)

    def test_interval_coverage_all_inside(self):
        y = np.array([2.0, 3.0, 4.0])
        lower = np.array([1.0, 2.0, 3.0])
        upper = np.array([3.0, 4.0, 5.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_interval_coverage_none_inside(self):
        y = np.array([10.0, 10.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(0.0)

    def test_interval_width(self):
        lower = np.array([0.0, 0.0])
        upper = np.array([4.0, 6.0])
        assert interval_width(lower, upper) == pytest.approx(5.0)


class TestEvaluateForecaster:
    # Use a large enough dataset so that lag/rolling features (up to 168 h)
    # still leave plenty of rows in the test split.
    N = 2000

    def test_returns_dataframe_with_mae(self):
        df = make_price_series(self.N)
        from src.models.forecasting import time_series_split
        train_df, test_df = time_series_split(df, test_fraction=0.2)

        forecaster = ElectricityPriceForecaster(
            horizons=[1],
            quantiles=[0.1, 0.5, 0.9],
            lgbm_params={"n_estimators": 5, "verbose": -1, "n_jobs": 1},
        )
        forecaster.fit(train_df)
        metrics = evaluate_forecaster(forecaster, test_df, horizons=[1])
        assert "mae" in metrics.columns
        assert "rmse" in metrics.columns
        assert "interval_coverage" in metrics.columns
        assert metrics.loc[1, "mae"] >= 0

    def test_mae_finite(self):
        df = make_price_series(self.N)
        from src.models.forecasting import time_series_split
        train_df, test_df = time_series_split(df, test_fraction=0.2)

        forecaster = ElectricityPriceForecaster(
            horizons=[1],
            quantiles=[0.5],
            lgbm_params={"n_estimators": 5, "verbose": -1, "n_jobs": 1},
        )
        forecaster.fit(train_df)
        metrics = evaluate_forecaster(forecaster, test_df, horizons=[1])
        assert np.isfinite(metrics.loc[1, "mae"])
