"""
Tests for src/model_utils.py (preprocessing + evaluation helpers) and
src/autoencoder_utils.py (the PyTorch Autoencoder), used by both
train_models.py and app.py.

Run with:
    pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.autoencoder_utils import Autoencoder
from src.model_utils import (
    PCA_COLUMNS,
    RAW_FEATURE_ORDER,
    bootstrap_pr_auc_ci,
    build_model_pipeline,
    build_preprocessing_pipeline,
    cost_optimal_threshold,
    minmax,
)


def make_fake_raw_df(n=200, seed=0):
    """Small synthetic dataframe matching the real dataset's raw schema."""
    rng = np.random.RandomState(seed)
    data = {col: rng.randn(n) for col in PCA_COLUMNS}
    data["Amount"] = rng.exponential(50, n)
    data["Time"] = np.sort(rng.uniform(0, 100000, n))
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
class TestPreprocessingPipeline:
    def test_output_shape_matches_input_rows(self):
        df = make_fake_raw_df(n=50)
        pipeline = build_preprocessing_pipeline()
        out = pipeline.fit_transform(df)
        assert out.shape[0] == 50
        assert out.shape[1] == len(PCA_COLUMNS) + 2  # +Amount, +Time

    def test_pca_columns_pass_through_unchanged(self):
        df = make_fake_raw_df(n=30)
        pipeline = build_preprocessing_pipeline()
        out = pipeline.fit_transform(df)
        # First len(PCA_COLUMNS) output columns should equal the raw V1..V28 values
        np.testing.assert_allclose(out[:, :len(PCA_COLUMNS)], df[PCA_COLUMNS].values)

    def test_amount_and_time_are_scaled(self):
        df = make_fake_raw_df(n=500, seed=1)
        pipeline = build_preprocessing_pipeline()
        out = pipeline.fit_transform(df)
        amount_scaled = out[:, len(PCA_COLUMNS)]
        time_scaled = out[:, len(PCA_COLUMNS) + 1]
        # A StandardScaler output should have ~zero mean, ~unit variance
        assert abs(amount_scaled.mean()) < 0.05
        assert abs(amount_scaled.std() - 1.0) < 0.05
        assert abs(time_scaled.mean()) < 0.05
        assert abs(time_scaled.std() - 1.0) < 0.05

    def test_transform_on_new_data_uses_fitted_params_not_refit(self):
        """
        This is the exact bug class the Pipeline refactor was meant to kill:
        calling .transform() on new data must NOT refit the scaler on that
        new data's distribution.
        """
        df_train = make_fake_raw_df(n=1000, seed=2)
        df_new = make_fake_raw_df(n=5, seed=99)
        df_new["Amount"] = [1_000_000.0] * 5  # wildly out-of-distribution value

        pipeline = build_preprocessing_pipeline()
        pipeline.fit(df_train)
        out_new = pipeline.transform(df_new)

        amount_scaled_new = out_new[:, len(PCA_COLUMNS)]
        # A $1,000,000 transaction should scale to a large positive number
        # using TRAIN statistics, not land near zero because of being
        # (wrongly) refit on itself.
        assert amount_scaled_new[0] > 10

    def test_raw_feature_order_has_no_duplicates_and_right_length(self):
        assert len(RAW_FEATURE_ORDER) == len(set(RAW_FEATURE_ORDER))
        assert len(RAW_FEATURE_ORDER) == len(PCA_COLUMNS) + 2


# ---------------------------------------------------------------------------
# Full model pipeline (preprocessing + classifier)
# ---------------------------------------------------------------------------
class TestModelPipeline:
    def test_pipeline_fits_and_predicts(self):
        from sklearn.linear_model import LogisticRegression

        df = make_fake_raw_df(n=300, seed=3)
        y = pd.Series(np.random.RandomState(3).binomial(1, 0.1, size=300))

        pipeline = build_model_pipeline(LogisticRegression(max_iter=200))
        pipeline.fit(df, y)
        proba = pipeline.predict_proba(df)[:, 1]

        assert proba.shape == (300,)
        assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# Cost-optimal threshold
# ---------------------------------------------------------------------------
class TestCostOptimalThreshold:
    def test_returns_value_between_0_and_1(self):
        y_true = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.05, 0.9, 0.8, 0.3, 0.15, 0.6, 0.25, 0.05])
        t = cost_optimal_threshold(y_true, y_proba)
        assert 0.0 <= t <= 1.0

    def test_high_fn_cost_favors_lower_threshold(self):
        """
        If missing fraud is made extremely expensive relative to a false
        alarm, the optimal threshold should move DOWN (catch more fraud,
        tolerate more false alarms) compared to a near-equal cost setup.
        """
        rng = np.random.RandomState(0)
        y_true = rng.binomial(1, 0.1, size=500)
        y_proba = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, size=500), 0, 1)

        t_high_fn_cost = cost_optimal_threshold(y_true, y_proba, cost_fn=10000, cost_fp=1)
        t_balanced = cost_optimal_threshold(y_true, y_proba, cost_fn=5, cost_fp=5)

        assert t_high_fn_cost <= t_balanced

    def test_return_curve_lengths_match(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        y_proba = np.linspace(0, 1, 10)
        best_t, thresholds, costs = cost_optimal_threshold(y_true, y_proba, return_curve=True)
        assert len(thresholds) == len(costs)
        assert 0.0 <= best_t <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    def test_ci_bounds_are_ordered_and_contain_point_estimate_roughly(self):
        from sklearn.metrics import average_precision_score

        rng = np.random.RandomState(0)
        y_true = rng.binomial(1, 0.15, size=400)
        y_proba = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, size=400), 0, 1)

        point_estimate = average_precision_score(y_true, y_proba)
        lo, hi = bootstrap_pr_auc_ci(y_true, y_proba, n_boot=200, random_state=0)

        assert lo <= hi
        # Point estimate should typically fall inside or very near the interval
        assert lo - 0.1 <= point_estimate <= hi + 0.1

    def test_narrower_ci_with_more_data(self):
        """More test data -> tighter confidence interval, as expected."""
        rng = np.random.RandomState(1)

        def make(n):
            y_true = rng.binomial(1, 0.15, size=n)
            y_proba = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, size=n), 0, 1)
            return y_true, y_proba

        y_small, p_small = make(100)
        y_large, p_large = make(5000)

        lo_s, hi_s = bootstrap_pr_auc_ci(y_small, p_small, n_boot=200, random_state=1)
        lo_l, hi_l = bootstrap_pr_auc_ci(y_large, p_large, n_boot=200, random_state=1)

        assert (hi_l - lo_l) < (hi_s - lo_s)


# ---------------------------------------------------------------------------
# minmax helper
# ---------------------------------------------------------------------------
class TestMinMax:
    def test_scales_to_0_1_range(self):
        arr = np.array([5, 10, 15, 20])
        out = minmax(arr)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_constant_array_returns_zeros_not_nan(self):
        arr = np.array([7, 7, 7, 7])
        out = minmax(arr)
        assert not np.isnan(out).any()
        assert (out == 0).all()


# ---------------------------------------------------------------------------
# Autoencoder architecture
# ---------------------------------------------------------------------------
class TestAutoencoder:
    def test_output_shape_matches_input(self):
        model = Autoencoder(input_dim=30)
        x = torch.randn(16, 30)
        out = model(x)
        assert out.shape == x.shape

    def test_bottleneck_is_smaller_than_input(self):
        model = Autoencoder(input_dim=30)
        # Bottleneck layer (last layer of encoder before ReLU) should be 7-dim
        bottleneck_layer = model.encoder[-2]
        assert bottleneck_layer.out_features == 7