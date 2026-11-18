from datetime import datetime, timezone

import pytest

from dataset.config import ALL_FEATURES
from dataset.phase7c_contract import (
    CANONICAL_FEATURES_42,
    AnalysisMetadataRecord,
    ContractError,
    ExtractionStatus,
    Feature42Record,
    RepositoryIdentity,
    SampleRecord,
    validate_feature_columns,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_feature_contract_is_derived_and_exactly_42() -> None:
    assert CANONICAL_FEATURES_42 == tuple(ALL_FEATURES)
    assert len(CANONICAL_FEATURES_42) == 42


def test_feature_order_extra_and_missing_columns_are_rejected() -> None:
    values = {name: 1.0 for name in CANONICAL_FEATURES_42}
    Feature42Record("s1", "v1", values)
    with pytest.raises(ContractError):
        validate_feature_columns(tuple(reversed(CANONICAL_FEATURES_42)))
    with pytest.raises(ContractError):
        Feature42Record("s1", "v1", {**values, "merged_at": 1.0})
    with pytest.raises(ContractError):
        Feature42Record("s1", "v1", {name: 1.0 for name in CANONICAL_FEATURES_42[:-1]})


def test_post_merge_and_label_fields_cannot_become_features() -> None:
    for forbidden in ("merge_state", "review_outcome", "ci_after_t0", "label", "label_confidence", "evidence"):
        with pytest.raises(ContractError):
            validate_feature_columns(CANONICAL_FEATURES_42 + (forbidden,))


def test_mixed_language_and_unsupported_coverage_are_explicit() -> None:
    record = AnalysisMetadataRecord(
        sample_id="s1",
        changed_languages=("javascript", "python"),
        primary_changed_language="python",
        changed_file_language_counts={"python": 2, "javascript": 1},
        changed_line_language_shares={"python": 0.8, "javascript": 0.2},
        repository_primary_language="python",
        adapter_versions={"python": "7.0"},
        adapter_statuses={"python": "validated", "javascript": "unsupported"},
        supported_language_coverage=0.8,
        unsupported_languages=("javascript",),
        extraction_status=ExtractionStatus.PARTIAL,
    )
    assert record.changed_languages == ("javascript", "python")
    assert record.unsupported_languages == ("javascript",)


def test_sample_carries_network_and_related_commit_provenance() -> None:
    sample = SampleRecord(
        "s1", RepositoryIdentity("github", "repo-1", "o/r", "network-1", "fork-1"),
        "pr-1", 1, "base", "head", "merge", NOW, NOW, "api-v1",
        repeated_commit_group_id="commit-group", patch_identity="patch-hash", related_event_group_id="event-group",
    )
    assert sample.repository.repository_network_id == "network-1"
    assert sample.patch_identity == "patch-hash"
