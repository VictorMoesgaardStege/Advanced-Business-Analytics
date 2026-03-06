"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_price_trend_features,
    build_feature_matrix,
    get_feature_columns,
)
from tests.conftest import make_price_series


class TestCalendarFeatures:
    def test_columns_present(self):
        df = make_price_series(100)
        out = add_calendar_features(df)
        expected_cols = {
            "hour", "day_of_week", "month", "year", "is_weekend",
            "is_danish_holiday", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
        }
        assert expected_cols.issubset(set(out.columns))

    def test_hour_range(self):
        df = make_price_series(100)
        out = add_calendar_features(df)
        assert out["hour"].between(0, 23).all()

    def test_cyclical_encoding_bounds(self):
        df = make_price_series(100)
        out = add_calendar_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert out[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

    def test_is_weekend_dtype(self):
        df = make_price_series(200)
        out = add_calendar_features(df)
        assert set(out["is_weekend"].unique()).issubset({0, 1})


class TestLagFeatures:
    def test_lag_columns_created(self):
        df = make_price_series(200)
        out = add_lag_features(df, lags=[1, 24])
        assert "lag_1h" in out.columns
        assert "lag_24h" in out.columns

    def test_lag_values_correct(self):
        df = make_price_series(200)
        out = add_lag_features(df, lags=[1])
        # lag_1h at row i should equal SpotPriceDKK at row i-1
        assert np.isnan(out["lag_1h"].iloc[0])
        assert out["lag_1h"].iloc[1] == pytest.approx(df["SpotPriceDKK"].iloc[0])

    def test_no_mutation(self):
        df = make_price_series(100)
        original_cols = list(df.columns)
        add_lag_features(df)
        assert list(df.columns) == original_cols


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = make_price_series(200)
        out = add_rolling_features(df, windows=[6])
        for col in ["rolling_mean_6h", "rolling_std_6h", "rolling_min_6h", "rolling_max_6h"]:
            assert col in out.columns

    def test_rolling_mean_non_negative(self):
        df = make_price_series(200)
        out = add_rolling_features(df, windows=[6])
        valid = out["rolling_mean_6h"].dropna()
        assert (valid >= 0).all()


class TestBuildFeatureMatrix:
    def test_target_column_present(self):
        df = make_price_series(500)
        feat_df = build_feature_matrix(df, forecast_horizon=1)
        assert "y" in feat_df.columns

    def test_no_nans_after_drop(self):
        df = make_price_series(500)
        feat_df = build_feature_matrix(df, forecast_horizon=1, drop_na=True)
        assert feat_df.isnull().sum().sum() == 0

    def test_target_equals_future_price(self):
        df = make_price_series(500)
        feat_df = build_feature_matrix(df, forecast_horizon=1, drop_na=False)
        # At each row i, y should equal SpotPriceDKK at i+1
        for i in range(len(df) - 1):
            expected_y = df["SpotPriceDKK"].iloc[i + 1]
            actual_y = feat_df["y"].iloc[i]
            if not (np.isnan(expected_y) or np.isnan(actual_y)):
                assert actual_y == pytest.approx(expected_y)

    def test_horizon_affects_target(self):
        df = make_price_series(500)
        feat_1h = build_feature_matrix(df, forecast_horizon=1, drop_na=True)
        feat_24h = build_feature_matrix(df, forecast_horizon=24, drop_na=True)
        # Different horizons → different targets
        assert not feat_1h["y"].equals(feat_24h["y"])


class TestGetFeatureColumns:
    def test_excludes_target_and_price(self):
        df = make_price_series(500)
        feat_df = build_feature_matrix(df, forecast_horizon=1)
        cols = get_feature_columns(feat_df)
        assert "y" not in cols
        assert "SpotPriceDKK" not in cols

    def test_returns_list(self):
        df = make_price_series(500)
        feat_df = build_feature_matrix(df, forecast_horizon=1)
        cols = get_feature_columns(feat_df)
        assert isinstance(cols, list)
        assert len(cols) > 0
