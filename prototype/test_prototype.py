"""
Fast checks of the prototype's detector (a few seconds, no server needed).

Run from the project root:  python -m unittest discover prototype -v
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import (Detector, WrongPassword, PASSWORD_KEYS, MIN_ENROL_ATTEMPTS,
                      pair_key_events, features_from_timestamps)
from downsample import downsample_row, _reconstruct_timestamps, _quantize


def key_events(press, release):
    """Wire-format events (ms) for one clean password entry."""
    events = [(key, "down", press[i] * 1000) for i, key in enumerate(PASSWORD_KEYS)]
    events += [(key, "up", release[i] * 1000) for i, key in enumerate(PASSWORD_KEYS)]
    return sorted(events, key=lambda event: event[2])


class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles_dir = tempfile.mkdtemp()
        cls.detector = Detector(profiles_dir=cls.profiles_dir)
        cls.cols = cls.detector.feature_cols
        cls.row = cls.detector.data.iloc[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.profiles_dir)

    def test_timestamps_round_trip_to_the_dataset_features(self):
        press, release = _reconstruct_timestamps(self.row)
        built = features_from_timestamps(press, release, poll_ms=1)
        # the dataset stores UD rounded to 4 decimals, so allow that much
        np.testing.assert_allclose(built[self.cols].to_numpy()[0],
                                   self.row[self.cols].to_numpy(dtype=float), atol=2e-4)

    def test_client_quantisation_matches_the_study_downsampling(self):
        dt = 0.030
        expected = downsample_row(self.row, dt, np.random.default_rng(7))
        phase = np.random.default_rng(7).uniform(0, dt)
        press, release = _reconstruct_timestamps(self.row)
        built = features_from_timestamps(_quantize(press, dt, phase), _quantize(release, dt, phase), 30)
        np.testing.assert_allclose(built[self.cols].to_numpy()[0],
                                   expected[self.cols].to_numpy(dtype=float), atol=1e-12)

    def test_key_events_pair_back_to_timestamps(self):
        press, release = _reconstruct_timestamps(self.row)
        got_press, got_release = pair_key_events(key_events(press, release))
        np.testing.assert_allclose(got_press, press)
        np.testing.assert_allclose(got_release, release)

    def test_wrong_or_incomplete_password_is_rejected(self):
        press, release = _reconstruct_timestamps(self.row)
        events = key_events(press, release)
        typo = [("x" if key == "t" else key, kind, t) for key, kind, t in events]
        corrected = events + [("Backspace", "down", 9999.0), ("Backspace", "up", 9999.5)]
        for bad in (typo, corrected, events[:-1], []):
            with self.assertRaises(WrongPassword):
                pair_key_events(bad)

    def test_holder_scores_below_impostors_and_alerts_follow_the_threshold(self):
        subject = self.detector.subjects[0]
        profile = self.detector.profile(subject)
        genuine, impostor = self.detector.replay_pool(subject)
        self.assertLess(profile.score(genuine).mean(), profile.score(impostor).mean())

        press, release = _reconstruct_timestamps(impostor.iloc[0])
        result = self.detector.assess(subject, press, release, poll_ms=1)
        self.assertEqual(result["alert"], result["score"] > result["threshold"])

    def test_threshold_is_the_percentile_of_enrolment_scores_and_rises_with_polling(self):
        profile = self.detector.profile(self.detector.subjects[0])
        self.assertAlmostEqual(profile.threshold(1), np.percentile(profile.train_scores, 90))
        self.assertGreater(profile.threshold(100), profile.threshold(1))

    def test_live_enrolment_creates_a_usable_account(self):
        rows = self.detector.data[self.detector.data["subject"] == self.detector.subjects[1]]
        for i in range(MIN_ENROL_ATTEMPTS):
            self.assertNotIn("demo", self.detector.accounts())
            self.detector.add_enrolment("demo", *_reconstruct_timestamps(rows.iloc[i]))
        self.assertIn("demo", self.detector.accounts())
        result = self.detector.assess("demo", *_reconstruct_timestamps(rows.iloc[50]), poll_ms=20)
        self.assertIn("alert", result)

        self.detector.reset_enrolment("demo")
        self.assertNotIn("demo", self.detector.accounts())
        with self.assertRaises(KeyError):
            self.detector.profile("demo")

    def test_bad_account_names_cannot_be_enrolled(self):
        press, release = _reconstruct_timestamps(self.row)
        for name in ("../evil", "", self.detector.subjects[0]):
            with self.assertRaises(ValueError):
                self.detector.add_enrolment(name, press, release)


if __name__ == "__main__":
    unittest.main()
