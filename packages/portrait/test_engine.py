"""Synthetic checks for anomaly rules and preset scales."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portrait.engine import _anomalies, _high, _zstats


def test_anomaly_finds_a_three_second_spike():
    values = [0.0] * 600
    for index in range(200, 230):
        values[index] = 8.0
    series = {"f0_hz": values}
    stats = {"f0_hz": _zstats(values)}
    rules = {
        "window_sec": 5,
        "min_duration_sec": 2,
        "z_threshold": 2,
        "min_baseline_sec": 30,
        "markers": ["f0_hz"],
    }
    found = _anomalies(series, stats, rules)
    assert found, "expected a spike to cross 2σ for at least 2 seconds"
    assert any(item["start_ms"] < 23000 and item["end_ms"] > 20000 for item in found)


def test_scale_change_moves_the_score():
    assert _high(14, 28) == 50
    assert _high(14, 14) == 100


if __name__ == "__main__":
    test_anomaly_finds_a_three_second_spike()
    test_scale_change_moves_the_score()
    print("ok")
