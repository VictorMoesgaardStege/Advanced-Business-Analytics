"""Tests for the decision support / recommendation module."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.recommendations.decision_support import (
    Recommendation,
    generate_recommendations,
    recommendations_to_dataframe,
    summarise_forecast,
)


def make_forecast_df(
    horizons=(1, 6, 12, 24, 48, 72, 168),
    base_price: float = 400.0,
    cheap_horizon: int = 24,
    cheap_price: float = 150.0,
    expensive_horizon: int = 6,
    expensive_price: float = 800.0,
) -> pd.DataFrame:
    """Create a synthetic forecast DataFrame."""
    rows = []
    for h in horizons:
        if h == cheap_horizon:
            med = cheap_price
        elif h == expensive_horizon:
            med = expensive_price
        else:
            med = base_price
        rows.append({"horizon_h": h, "q0.10": med * 0.8, "q0.50": med, "q0.90": med * 1.2})
    return pd.DataFrame(rows).set_index("horizon_h")


class TestSummariseForecast:
    def test_keys_present(self):
        fc = make_forecast_df()
        summary = summarise_forecast(fc)
        assert "min_median" in summary
        assert "max_median" in summary
        assert "cheapest_horizon_h" in summary
        assert "most_expensive_horizon_h" in summary
        assert "price_range_dkk" in summary

    def test_cheapest_horizon_correct(self):
        fc = make_forecast_df(cheap_horizon=24, cheap_price=100.0)
        summary = summarise_forecast(fc)
        assert summary["cheapest_horizon_h"] == 24

    def test_price_range_correct(self):
        fc = make_forecast_df(
            cheap_price=100.0, cheap_horizon=24,
            expensive_price=900.0, expensive_horizon=6,
        )
        summary = summarise_forecast(fc)
        assert summary["price_range_dkk"] == pytest.approx(800.0)


class TestGenerateRecommendations:
    def test_returns_list(self):
        fc = make_forecast_df()
        recs = generate_recommendations(fc, current_price_dkk=400.0)
        assert isinstance(recs, list)
        assert len(recs) >= 1

    def test_saving_message_when_price_drops(self):
        fc = make_forecast_df(cheap_horizon=24, cheap_price=150.0)
        recs = generate_recommendations(fc, current_price_dkk=600.0)
        messages = [r.message for r in recs]
        assert any("wait" in m.lower() or "drop" in m.lower() for m in messages)

    def test_expensive_recommendation_present(self):
        fc = make_forecast_df(expensive_horizon=6, expensive_price=900.0)
        recs = generate_recommendations(fc, current_price_dkk=400.0)
        categories = [r.category for r in recs]
        assert "expensive" in categories

    def test_recommendation_fields_populated(self):
        fc = make_forecast_df()
        recs = generate_recommendations(fc, current_price_dkk=500.0)
        for rec in recs:
            assert rec.horizon_h is not None
            assert rec.category in {"cheap", "normal", "expensive"}
            assert isinstance(rec.message, str)
            assert len(rec.message) > 10

    def test_no_negative_savings(self):
        fc = make_forecast_df()
        recs = generate_recommendations(fc, current_price_dkk=100.0)
        for rec in recs:
            if rec.estimated_saving_dkk_per_mwh is not None:
                assert rec.estimated_saving_dkk_per_mwh >= 0


class TestRecommendationsToDataFrame:
    def test_returns_dataframe(self):
        fc = make_forecast_df()
        recs = generate_recommendations(fc, current_price_dkk=400.0)
        df = recommendations_to_dataframe(recs)
        assert isinstance(df, pd.DataFrame)
        assert "message" in df.columns
        assert "category" in df.columns
        assert len(df) == len(recs)
