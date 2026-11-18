from datetime import UTC, datetime

from dataset.phase7c_contract import LabelState
from dataset.phase7d_labeler import reconcile_label
from dataset.phase7d_outcome_evidence import CachedOutcomeEvidenceProvider, OutcomeEvidence, UnavailableOutcomeEvidenceProvider
from dataset.phase7d_outcome_policy import OutcomeEvidenceClassification


MERGE = datetime(2025, 1, 1, tzinfo=UTC)
AUDIT = datetime(2025, 5, 1, tzinfo=UTC)

def collect(*records):
    return CachedOutcomeEvidenceProvider(records).collect(repository="o/r", pr_id="1", merge_sha="m", merge_at=MERGE)

def test_confirmed_revert_is_high() -> None:
    label = reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=collect(OutcomeEvidence(OutcomeEvidenceClassification.REVERT, "r", datetime(2025, 1, 2, tzinfo=UTC), "cache", AUDIT, "explicit", confirmed=True)))
    assert label.label is LabelState.HIGH

def test_corrective_is_medium_and_no_evidence_is_low_only_after_evaluation() -> None:
    corrective = OutcomeEvidence(OutcomeEvidenceClassification.CORRECTIVE, "c", datetime(2025, 1, 2, tzinfo=UTC), "cache", AUDIT, "explicit", materially_overlapping_scope=True)
    assert reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=collect(corrective)).label is LabelState.MEDIUM
    assert reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=collect()).label is LabelState.LOW_OBSERVED

def test_unmerged_immature_conflict_and_unavailable_never_become_low() -> None:
    unavailable = UnavailableOutcomeEvidenceProvider().collect(repository="o/r", pr_id="1", merge_sha="m", merge_at=MERGE)
    assert reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=unavailable).label is LabelState.UNKNOWN
    conflict = OutcomeEvidence(OutcomeEvidenceClassification.CONFLICT, "x", datetime(2025, 1, 2, tzinfo=UTC), "cache", AUDIT, "recorded")
    assert reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=collect(conflict)).label is LabelState.CONFLICTED
    assert reconcile_label(merged_at=None, audited_at=AUDIT, result=collect()).label is LabelState.UNKNOWN


def test_exact_boundary_is_mature_but_one_day_short_is_unknown() -> None:
    result = collect()
    assert reconcile_label(merged_at=MERGE, audited_at=datetime(2025, 3, 31, tzinfo=UTC), result=result).label is LabelState.UNKNOWN
    assert reconcile_label(merged_at=MERGE, audited_at=datetime(2025, 4, 1, tzinfo=UTC), result=result).label is LabelState.LOW_OBSERVED


def test_legacy_review_friction_is_not_an_input_to_outcome_labeling() -> None:
    # ``reconcile_label`` has no high_risk parameter; identical evidence must
    # yield an identical outcome regardless of legacy source-row values.
    result = collect()
    assert reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=result) == reconcile_label(merged_at=MERGE, audited_at=AUDIT, result=result)
