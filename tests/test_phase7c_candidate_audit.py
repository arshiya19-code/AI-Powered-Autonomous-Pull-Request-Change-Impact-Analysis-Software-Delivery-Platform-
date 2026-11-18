from datetime import datetime, timezone

import pytest

from dataset.phase7c_candidate_audit import CandidateRecord, audit_candidates
from dataset.phase7c_contract import (
    AnalysisMetadataRecord, ExtractionStatus, LabelRecord, LabelState, QualityRecord, RepositoryIdentity, SampleRecord,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candidate(sample_id: str, *, language: str = "python", coverage: float = 1.0, included: bool = True,
              status: ExtractionStatus = ExtractionStatus.SUCCESS, label: LabelState = LabelState.LOW_OBSERVED,
              reason: str | None = None, network: str = "network-1", lines: int = 10) -> CandidateRecord:
    return CandidateRecord(
        SampleRecord(sample_id, RepositoryIdentity("github", f"repo-{sample_id}", f"o/r{sample_id}", network),
                     f"pr-{sample_id}", 1, f"base-{sample_id}", f"head-{sample_id}", "merge", NOW, NOW, "v1"),
        AnalysisMetadataRecord(sample_id, (language,), language, {language: 1}, {language: 1.0}, language,
                               {language: "v1"}, {language: "validated"}, coverage,
                               () if coverage == 1 else (language,), status),
        LabelRecord(sample_id, label, "high", "none", (), NOW, NOW, "v1"),
        QualityRecord(sample_id, included, reason, None),
        lines, 1, "small", "active", "2025-Q1",
    )


def test_audit_reports_natural_prevalence_and_reconciles_attrition() -> None:
    report = audit_candidates([
        candidate("1", label=LabelState.HIGH),
        candidate("2", language="javascript", coverage=0.0, included=False, reason="unsupported_language"),
        candidate("3", included=False, reason="parse_failed", status=ExtractionStatus.FAILED),
    ])
    assert report.availability["candidates_discovered"] == 3
    assert report.availability["eligible_analyzable"] + report.availability["candidates_excluded"] == 3
    assert sum(report.exclusion_reasons.values()) == report.availability["candidates_excluded"]
    assert report.label_prevalence == {"high": 1, "low_observed": 2}
    assert report.natural_language_distribution == {"javascript": 1, "python": 2}
    assert report.eligible_language_distribution == {"python": 1}
    assert report.balancing_performed is False


def test_audit_rejects_duplicate_repository_pr_identity() -> None:
    first = candidate("1")
    duplicate = candidate("2")
    duplicate = CandidateRecord(
        SampleRecord("2", first.sample.repository, first.sample.pr_id, 1, "base", "head", "merge", NOW, NOW, "v1"),
        duplicate.analysis, duplicate.label, duplicate.quality, 3, 1,
    )
    with pytest.raises(ValueError, match="duplicate PR IDs"):
        audit_candidates([first, duplicate])


def test_repository_and_label_strata_are_reported() -> None:
    report = audit_candidates([candidate("1", network="n1", lines=1), candidate("2", network="n2", lines=100)])
    assert report.repository_distribution["repository_network_count"] == 2
    assert report.label_prevalence_by_language["python"]["low_observed"] == 2
    assert sum(report.pr_size_distribution["changed_line_strata"].values()) == 2
