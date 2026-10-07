"""
Fast checks of the pipeline's building blocks (no model training, ~10 seconds).

Run from the project root:  python -m unittest discover tests -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eer import compute_eer
from downsample import downsample_dataframe, downsample_row, _h_cols, _dd_cols, _ud_cols
from subject_models import (SubjectModels, N_TRAIN_REPS, N_IMPOSTOR_REPS, training_reps,
                            split_subject, get_feature_columns, rows_to_sequences,
                            zscore, fuse_scores)
from hypothesis_verdicts import bootstrap_ci, holm, signed_rank_p
from cost_projection import project_costs, KEY_EVENTS_PER_PASSWORD

RAW_CSV = os.path.join(ROOT, "data", "raw", "DSL-StrongPasswordData.csv")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


class TestEER(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_matches_analytical_value_for_shifted_normals(self):
        # unit-variance normals d apart cross at the midpoint: EER = Phi(-d/2)
        for d in [1.0, 2.0, 3.0]:
            result = compute_eer(self.rng.normal(0, 1, 50_000), self.rng.normal(d, 1, 50_000))
            self.assertAlmostEqual(result["eer"], norm.cdf(-d / 2), delta=0.005)
            self.assertAlmostEqual(result["threshold"], d / 2, delta=0.05)

    def test_identical_distributions_give_chance(self):
        result = compute_eer(self.rng.normal(0, 1, 50_000), self.rng.normal(0, 1, 50_000))
        self.assertAlmostEqual(result["eer"], 0.5, delta=0.01)

    def test_perfect_separation_gives_zero(self):
        result = compute_eer(self.rng.normal(0, 0.1, 1000), self.rng.normal(10, 0.1, 1000))
        self.assertLess(result["eer"], 0.001)

    def test_invert_scores_is_equivalent_to_negating(self):
        g, i = self.rng.normal(0, 1, 5000), self.rng.normal(2, 1, 5000)
        self.assertAlmostEqual(compute_eer(-g, -i, invert_scores=True)["eer"],
                               compute_eer(g, i)["eer"], places=12)


class TestDownsampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw().head(600)

    def test_control_interval_leaves_data_unchanged(self):
        pd.testing.assert_frame_equal(downsample_dataframe(self.df, dt=0.001), self.df)

    def test_vectorised_matches_row_by_row_reference(self):
        rng = np.random.default_rng(7)
        reference = self.df.apply(lambda r: downsample_row(r, 0.020, rng), axis=1)
        fast = downsample_dataframe(self.df, dt=0.020, seed=7)
        cols = _h_cols() + _dd_cols() + _ud_cols()
        np.testing.assert_array_equal(fast[cols].to_numpy(float), reference[cols].to_numpy(float))

    def test_intervals_are_quantised_and_floored(self):
        for dt in [0.020, 0.030, 0.100]:
            out = downsample_dataframe(self.df, dt=dt, seed=1)
            for cols in (_h_cols(), _dd_cols()):
                v = out[cols].to_numpy()
                self.assertGreaterEqual(v.min(), dt / 2 - 1e-12)
                # every interval is a whole number of ticks, or the half-tick floor
                half_ticks = v / (dt / 2)
                np.testing.assert_allclose(half_ticks, np.round(half_ticks), atol=1e-6)

    def test_up_down_latency_stays_consistent(self):
        out = downsample_dataframe(self.df, dt=0.030, seed=1)
        expected = out[_dd_cols()].to_numpy() - out[_h_cols()].to_numpy()[:, :-1]
        np.testing.assert_allclose(out[_ud_cols()].to_numpy(), expected, atol=1e-12)

    def test_seed_controls_the_jitter(self):
        a = downsample_dataframe(self.df, dt=0.020, seed=3)
        b = downsample_dataframe(self.df, dt=0.020, seed=3)
        c = downsample_dataframe(self.df, dt=0.020, seed=4)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a[_h_cols()].equals(c[_h_cols()]))

    def test_distortion_grows_with_interval(self):
        distortion = [(downsample_dataframe(self.df, dt=dt, seed=0)["H.t"] - self.df["H.t"]).abs().mean()
                      for dt in [0.020, 0.030, 0.100]]
        self.assertEqual(distortion, sorted(distortion))


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw()
        cls.subject = sorted(cls.df["subject"].unique())[0]

    def test_training_reps_are_first_four_sessions(self):
        train = training_reps(self.df, self.subject)
        self.assertEqual(len(train), N_TRAIN_REPS)
        self.assertEqual(set(train["subject"]), {self.subject})
        self.assertEqual(sorted(train["sessionIndex"].unique()), [1, 2, 3, 4])

    def test_genuine_test_reps_do_not_overlap_training(self):
        train = training_reps(self.df, self.subject)
        genuine, _ = split_subject(self.df, self.subject)
        self.assertEqual(len(genuine), 400 - N_TRAIN_REPS)
        self.assertEqual(set(genuine["subject"]), {self.subject})
        self.assertEqual(sorted(genuine["sessionIndex"].unique()), [5, 6, 7, 8])
        self.assertFalse(set(train.index) & set(genuine.index))

    def test_impostors_are_first_reps_of_every_other_subject(self):
        _, impostor = split_subject(self.df, self.subject)
        n_others = self.df["subject"].nunique() - 1
        self.assertEqual(len(impostor), n_others * N_IMPOSTOR_REPS)
        self.assertNotIn(self.subject, set(impostor["subject"]))
        self.assertTrue((impostor.groupby("subject").size() == N_IMPOSTOR_REPS).all())
        self.assertEqual(set(impostor["sessionIndex"]), {1})

    def test_sequence_framing(self):
        rows = self.df.head(5)
        seq = rows_to_sequences(rows)
        self.assertEqual(seq.shape, (5, 11, 3))
        self.assertEqual(len(get_feature_columns(self.df)), 31)
        # first key has no incoming latency; later timesteps carry the matching columns
        np.testing.assert_array_equal(seq[:, 0, 1:], 0)
        np.testing.assert_array_equal(seq[:, 1, 0], rows["H.t"].to_numpy())
        np.testing.assert_array_equal(seq[:, 1, 1], rows["DD.period.t"].to_numpy())
        np.testing.assert_array_equal(seq[:, 1, 2], rows["UD.period.t"].to_numpy())


class TestFusion(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.models = SubjectModels([], None, None, None,
                                    if_train_scores=rng.normal(0.4, 0.05, 200),
                                    lstm_train_scores=rng.normal(1.0, 0.3, 200))
        self.if_scores = rng.normal(0.5, 0.05, 50)
        self.lstm_scores = rng.normal(1.5, 0.3, 50)

    def test_extreme_weights_reduce_to_single_models(self):
        np.testing.assert_allclose(
            fuse_scores(self.models, self.if_scores, self.lstm_scores, alpha=1.0),
            zscore(self.models.if_train_scores, self.if_scores))
        np.testing.assert_allclose(
            fuse_scores(self.models, self.if_scores, self.lstm_scores, alpha=0.0),
            zscore(self.models.lstm_train_scores, self.lstm_scores))

    def test_zscore_uses_training_distribution(self):
        z = zscore(self.models.if_train_scores, self.models.if_train_scores)
        self.assertAlmostEqual(z.mean(), 0, places=10)
        self.assertAlmostEqual(z.std(), 1, places=10)

    def test_zscore_survives_constant_training_scores(self):
        self.assertTrue(np.isfinite(zscore(np.ones(10), np.array([1.0, 2.0]))).all())


class TestStatistics(unittest.TestCase):
    def test_holm_correction_known_values(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])

    def test_bootstrap_ci_brackets_the_mean_and_is_reproducible(self):
        x = np.random.default_rng(1).normal(1.0, 2.0, 51)
        lo, hi = bootstrap_ci(x)
        self.assertLess(lo, x.mean())
        self.assertGreater(hi, x.mean())
        self.assertEqual((lo, hi), bootstrap_ci(x))
        # close to the normal-theory interval for a sample this size
        half_width = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        self.assertAlmostEqual(hi - lo, 2 * half_width, delta=0.25 * half_width)

    def test_signed_rank_detects_a_shift_and_handles_all_zeros(self):
        rng = np.random.default_rng(2)
        self.assertLess(signed_rank_p(rng.normal(1.0, 1.0, 51)), 0.001)
        self.assertGreater(signed_rank_p(rng.normal(0.0, 1.0, 51)), 0.05)
        self.assertEqual(signed_rank_p(np.zeros(51)), 1.0)


class TestCostProjection(unittest.TestCase):
    def setUp(self):
        self.costs = project_costs(event_duration_sec=2.0, payload_bytes=100, n_events=1_000_000)

    def rows(self, architecture):
        return self.costs[self.costs["architecture"] == architecture].set_index("group")

    def test_sample_streaming_scales_with_interval(self):
        s = self.rows("sample_streaming")
        self.assertEqual(s.loc["control_1ms", "messages_per_event"], 2000)
        self.assertEqual(s.loc["test_20ms", "messages_per_event"], 100)
        # 2000M messages x $1/M + 33,333 connection-minutes x $0.25/M + (186.26 - 100) GB x $0.09
        self.assertAlmostEqual(s.loc["control_1ms", "total_monthly_cost_usd"], 2007.772, places=2)
        self.assertAlmostEqual(s.loc["test_20ms", "cost_savings_vs_1ms_pct"], 95.0, delta=0.1)

    def test_event_driven_cost_is_independent_of_interval(self):
        e = self.rows("event_driven")
        self.assertTrue((e["messages_per_event"] == KEY_EVENTS_PER_PASSWORD).all())
        self.assertEqual(KEY_EVENTS_PER_PASSWORD, 22)
        self.assertEqual(e["total_monthly_cost_usd"].nunique(), 1)
        self.assertTrue((e["cost_savings_vs_1ms_pct"] == 0).all())


if __name__ == "__main__":
    unittest.main()
