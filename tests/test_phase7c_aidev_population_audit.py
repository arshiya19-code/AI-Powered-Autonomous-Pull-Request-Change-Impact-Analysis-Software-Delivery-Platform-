from datetime import UTC, datetime

from dataset.phase7c_aidev_population_audit import audit_aidev_population


def test_population_audit_keeps_mature_pr_without_outcome_evidence_unknown() -> None:
    report = audit_aidev_population(
        [{"id": 1, "full_name": "org/repo", "language": "Python", "is_forked": False}],
        [{"id": 10, "number": 2, "repo_id": 1, "created_at": "2024-01-01T00:00:00Z", "merged_at": "2024-01-02T00:00:00Z"}],
        [{"pr_id": 10, "filename": "src/a.py", "changes": 12}],
        audited_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert report["population"]["merged"] == report["population"]["mature"] == 1
    assert report["population"]["labelable"] == 0
    assert report["labels"]["unknown"]["count"] == 1
    assert report["labels"]["low_observed"]["count"] == 0
    assert report["attrition"]["currently_fully_analyzable"] == 1


def test_population_audit_handles_unmerged_and_unsupported_language_without_balancing() -> None:
    report = audit_aidev_population(
        [{"id": 1, "full_name": "org/repo", "language": "Java"}],
        [{"id": 10, "number": 2, "repo_id": 1, "created_at": "2025-01-01T00:00:00Z", "merged_at": None}],
        [{"pr_id": 10, "filename": "src/A.java", "changes": 3}],
        audited_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert report["labels"]["unknown"]["count"] == 1
    assert report["languages"]["currently_fully_analyzable"] == 0
    assert report["reproducibility"]["seed"] == 42
