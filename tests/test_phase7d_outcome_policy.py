from datetime import UTC, datetime

from dataset.phase7d_outcome_policy import outcome_window


def test_outcome_window_is_fixed_at_90_days() -> None:
    window = outcome_window(datetime(2025, 1, 1, tzinfo=UTC))
    assert window.start.isoformat() == "2025-01-01T00:00:00+00:00"
    assert window.end.isoformat() == "2025-04-01T00:00:00+00:00"
