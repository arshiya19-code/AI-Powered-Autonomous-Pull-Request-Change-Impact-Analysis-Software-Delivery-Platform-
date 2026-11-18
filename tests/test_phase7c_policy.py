from datetime import datetime, timedelta, timezone

from dataset.phase7c_contract import LabelState
from dataset.phase7c_policy import (
    EvidenceKind,
    LabelEvidence,
    ObservationState,
    RepositoryCapPolicy,
    TemporalContext,
    classify_label,
    default_language_support_matrix,
    is_fully_analyzable,
    observation_state,
)


T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def context(**changes: object) -> TemporalContext:
    data: dict[str, object] = {
        "head_sha": "head", "head_committed_at": T0, "merge_sha": "merge", "merged_at": T0 + timedelta(hours=1),
        "analyzed_at": T0 + timedelta(hours=1, days=90),
    }
    data.update(changes)
    return TemporalContext(**data)  # type: ignore[arg-type]


def test_t0_and_window_are_respected() -> None:
    item = context(analyzed_at=T0 + timedelta(hours=1, days=89))
    assert item.t0_sha == "head"
    assert observation_state(item) is ObservationState.MERGED_IMMATURE
    assert observation_state(context(analyzed_at=T0 + timedelta(hours=1, days=90))) is ObservationState.MERGED_MATURE
    assert observation_state(context()) is ObservationState.MERGED_MATURE


def test_unknown_and_unmerged_never_become_low_observed() -> None:
    assert classify_label(context(analyzed_at=T0 + timedelta(days=1)), [], sufficient_no_adverse_evidence=True) is LabelState.UNKNOWN
    assert classify_label(context(merge_sha=None, merged_at=None), [], sufficient_no_adverse_evidence=True) is LabelState.UNKNOWN


def test_conflicted_high_medium_and_low_are_deterministic() -> None:
    mature = context()
    after_t0 = T0 + timedelta(days=2)
    assert classify_label(mature, [LabelEvidence(EvidenceKind.CONFLICT, "x", after_t0)]) is LabelState.CONFLICTED
    assert classify_label(mature, [LabelEvidence(EvidenceKind.CONFIRMED_ROLLBACK, "r", after_t0, confirmed=True)]) is LabelState.HIGH
    assert classify_label(mature, [LabelEvidence(EvidenceKind.CORRECTIVE_CHANGE, "c", after_t0, materially_overlapping_scope=True)]) is LabelState.MEDIUM
    assert classify_label(mature, [], sufficient_no_adverse_evidence=True) is LabelState.LOW_OBSERVED


def test_pre_t0_evidence_does_not_label_the_pr() -> None:
    assert classify_label(
        context(), [LabelEvidence(EvidenceKind.CONFIRMED_ROLLBACK, "old", T0, confirmed=True)],
        sufficient_no_adverse_evidence=True,
    ) is LabelState.LOW_OBSERVED


def test_pre_merge_and_post_window_evidence_do_not_count_as_outcomes() -> None:
    mature = context()
    pre_merge = T0 + timedelta(minutes=30)
    after_window = T0 + timedelta(hours=1, days=90, seconds=1)
    assert classify_label(
        mature,
        [LabelEvidence(EvidenceKind.CONFIRMED_ROLLBACK, "before-merge", pre_merge, confirmed=True)],
        sufficient_no_adverse_evidence=True,
    ) is LabelState.LOW_OBSERVED
    assert classify_label(
        mature,
        [LabelEvidence(EvidenceKind.CONFIRMED_ROLLBACK, "late", after_window, confirmed=True)],
        sufficient_no_adverse_evidence=True,
    ) is LabelState.LOW_OBSERVED


def test_validated_adapters_are_fully_analyzable() -> None:
    matrix = default_language_support_matrix()
    for language in ("python", "javascript", "typescript", "csharp", "rust", "cpp"):
        assert matrix[language].fully_analyzable
        assert is_fully_analyzable(language, matrix)
    assert not matrix["java"].fully_analyzable
    assert not is_fully_analyzable("java", matrix)


def test_repository_network_cap_and_shortage_are_deterministic() -> None:
    policy = RepositoryCapPolicy()
    assert policy.cap_for_selected_rows(100) == 2
    assert policy.cap_for_selected_rows(2_000) == 25
    assert policy.validate_network_availability({"python": 48}) == {"python": 2}
    assert policy.validate_network_caps({"network-a": 3, "network-b": 2}, 100) == ("network-a",)
    assert policy.deterministic_seed == 42
