import json
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.timer import StageTimer, load_metrics, timed, write_metrics


class TimerTests(unittest.TestCase):
    def test_stage_timer_accumulates_multiple_timings(self):
        timer = StageTimer()
        timer.record("train", "train_time_sec", 1.25)
        timer.record("train", "train_time_sec", 0.75)
        timer.record("train", "inference_time_sec", 0.5)

        self.assertAlmostEqual(timer.stages["train"]["train_time_sec"], 2.0)
        self.assertAlmostEqual(timer.stages["train"]["inference_time_sec"], 0.5)

    def test_timed_decorator_records_time(self):
        timer = StageTimer()
        perf_values = iter([10.0, 10.125])

        @timed("stage_1", "train_time_sec")
        def work(*, stage_timer=None):
            return "ok"

        with patch("code.utils.timer.time.perf_counter", side_effect=lambda: next(perf_values)):
            self.assertEqual(work(stage_timer=timer), "ok")

        self.assertAlmostEqual(timer.stages["stage_1"]["train_time_sec"], 0.125)

    def test_write_metrics_creates_valid_json_and_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics.json"
            initial = {
                "project": "COMP5434 Review Rating Prediction",
                "stages": {"0": {"rmse": None, "train_time_sec": None}},
            }
            write_metrics(str(output), initial)

            updated = write_metrics(
                str(output), {"stages": {"0": {"rmse": 0.91}, "1": {"rmse": 0.88}}}
            )

            loaded = load_metrics(str(output))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), loaded)
            self.assertEqual(loaded["stages"]["0"]["rmse"], 0.91)
            self.assertIsNone(loaded["stages"]["0"]["train_time_sec"])
            self.assertEqual(loaded["stages"]["1"]["rmse"], 0.88)
            self.assertEqual(updated, loaded)
