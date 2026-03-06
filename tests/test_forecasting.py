"""Tests for the forecasting models."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.forecasting import (
    ElectricityPriceForecaster,
    time_series_split,
    train_forecaster,
)
from tests.conftest import make_price_series


class TestTimeSeriesSplit:
    def test_sizes(self):
        df = make_price_series(1000)
        train, test = time_series_split(df, test_fraction=0.2)
        assert len(train) == 800
        assert len(test) == 200

    def test_temporal_ordering(self):
        df = make_price_series(1000)
        train, test = time_series_split(df, test_fraction=0.2)
        assert train.index.max() < test.index.min()

    def test_no_overlap(self):
        df = make_price_series(1000)
        train, test = time_series_split(df, test_fraction=0.2)
        assert len(set(train.index).intersection(set(test.index))) == 0


class TestElectricityPriceForecaster:
    @pytest.fixture(scope="class")
    def trained_forecaster(self):
        df = make_price_series(600)
        forecaster = ElectricityPriceForecaster(
            horizons=[1, 24],
            quantiles=[0.1, 0.5, 0.9],
            lgbm_params={
                "n_estimators": 10,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "n_jobs": 1,
                "random_state": 0,
                "verbose": -1,
            },
        )
        forecaster.fit(df)
        return forecaster

    def test_models_trained(self, trained_forecaster):
        assert 1 in trained_forecaster.models
        assert 24 in trained_forecaster.models

    def test_predict_returns_dataframe(self, trained_forecaster):
        df = make_price_series(600)
        forecast = trained_forecaster.predict_from_raw(df)
        assert not forecast.empty

    def test_predict_columns(self, trained_forecaster):
        df = make_price_series(600)
        forecast = trained_forecaster.predict_from_raw(df)
        assert "q0.10" in forecast.columns
        assert "q0.50" in forecast.columns
        assert "q0.90" in forecast.columns

    def test_predict_non_negative(self, trained_forecaster):
        df = make_price_series(600)
        forecast = trained_forecaster.predict_from_raw(df)
        assert (forecast >= 0).all().all()

    def test_quantile_ordering(self, trained_forecaster):
        """Lower quantile should generally be ≤ median ≤ upper."""
        df = make_price_series(600)
        forecast = trained_forecaster.predict_from_raw(df)
        assert (forecast["q0.10"] <= forecast["q0.50"] + 1e-6).all()
        assert (forecast["q0.50"] <= forecast["q0.90"] + 1e-6).all()

    def test_save_load_roundtrip(self, trained_forecaster, tmp_path):
        path = tmp_path / "test_forecaster.joblib"
        trained_forecaster.save(path)
        loaded = ElectricityPriceForecaster.load(path)
        assert loaded.horizons == trained_forecaster.horizons
        assert loaded.quantiles == trained_forecaster.quantiles

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ElectricityPriceForecaster.load(tmp_path / "nonexistent.joblib")

    def test_predict_before_fit_raises(self):
        from src.features.feature_engineering import build_feature_matrix
        forecaster = ElectricityPriceForecaster(horizons=[1])
        df = make_price_series(600)
        feat_df = build_feature_matrix(df, forecast_horizon=1, drop_na=True)
        with pytest.raises(RuntimeError, match="not been trained"):
            forecaster.predict(feat_df.iloc[[-1]])


class TestTrainForecasterConvenience:
    def test_returns_fitted_forecaster(self):
        df = make_price_series(600)
        forecaster = train_forecaster(
            df,
            horizons=[1],
            lgbm_params={"n_estimators": 5, "verbose": -1, "n_jobs": 1},
        )
        assert isinstance(forecaster, ElectricityPriceForecaster)
        assert 1 in forecaster.models
