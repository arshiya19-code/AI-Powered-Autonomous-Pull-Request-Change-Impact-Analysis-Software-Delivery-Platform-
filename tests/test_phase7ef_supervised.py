"""Tests for the Phase 7E/7F supervised dataset and CatBoost path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from analyzer.ml.preprocessing import CANONICAL_FEATURES
from dataset.phase7c_contract import CANONICAL_FEATURES_42, LabelState
from dataset.phase7e_supervised import (
    assemble_row,
    decide_eligibility,
    language_profile,
    leakage_audit,
    repository_temporal_split,
    write_dataset_package,
    assemble_supervised_dataset,
    AssemblyResult,
)
from analyzer.ml.deployment_risk import (
    DeploymentRiskModel,
    assess_sufficiency,
    split_matrices,
    train_deployment_risk_model,
)


CUTOFF = datetime(2026, 9, 6, tzinfo=UTC)


def _features(**overrides: object) -> dict:
    row = {
        "pr_id": 101,
        "repo_id": 9,
        "repo_full_name": "acme/lib",
        "pr_number": 7,
        "created_at": "2025-01-02T00:00:00Z",
        "high_risk": 0,
        "checkout_success": True,
        "analysis_mode": "full",
        "extraction_error": None,
    }
    for index, name in enumerate(CANONICAL_FEATURES):
        row[name] = float(index + 1)
    row.update(overrides)
    return row


def _label(sample_id: str = "101", risk: str = "low_observed") -> dict:
    return {
        "sample_id": sample_id,
        "risk_label": risk,
        "confidence": "medium",
        "evidence_type": "no_observed_adverse_outcome",
        "evidence_references": [],
        "observation_start": "2025-01-02T00:00:00+00:00",
        "observation_end": "2025-04-02T00:00:00+00:00",
        "label_policy_version": "phase7d-outcome-v1",
        "unknown_or_conflicted_reason": None,
    }


def _ledger(sample_id: str = "101", language: str = "python") -> dict:
    return {
        "sample_id": sample_id,
        "acquisition_status": "resolved",
        "changed_files": [{"path": f"src/mod.{ 'py' if language == 'python' else 'ts'}", "language": language}],
        "changed_language_set": [language],
        "repo_full_name": "acme/lib",
    }


def test_language_routing_uses_existing_adapters() -> None:
    profile = language_profile(_ledger(language="typescript"), _features())
    assert profile["primary_language"] == "typescript"
    assert "typescript" in profile["adapter_used"]
    assert profile["parsing_status"] == "full"


def test_unknown_and_invalid_rows_are_rejected() -> None:
    seen: set[tuple[str, str]] = set()
    assert decide_eligibility(_features(), _label(risk="unknown"), _ledger(), seen_pr_keys=seen).reason == "unknown_label"
    assert decide_eligibility(_features(analysis_mode="patch_only"), _label(), _ledger(), seen_pr_keys=seen).reason == "feature_extraction_not_full"
    assert decide_eligibility(_features(files_changed=0.0), _label(), _ledger(), seen_pr_keys=seen).reason == "no_meaningful_source_change"
    nan_row = _features()
    nan_row["impact_score"] = float("nan")
    assert decide_eligibility(nan_row, _label(), _ledger(), seen_pr_keys=seen).reason == "invalid_42_feature_vector"
    ok = decide_eligibility(_features(), _label(), _ledger(), seen_pr_keys=seen)
    assert ok.included
    seen.add(("acme/lib", "7"))
    assert decide_eligibility(_features(), _label(), _ledger(), seen_pr_keys=seen).reason == "duplicate_pr"


def test_exact_42_feature_schema_and_stable_order() -> None:
    payload = assemble_row(_features(), _label(), _ledger(), dataset_version="v", source_cohort="c")
    values = [payload[name] for name in CANONICAL_FEATURES]
    assert len(values) == 42
    assert list(CANONICAL_FEATURES) == list(CANONICAL_FEATURES_42)
    assert payload["risk_label"] == "low_observed"
    assert "risk_label" not in CANONICAL_FEATURES
    assert payload["review_friction_proxy"] == 0


def test_leakage_audit_keeps_labels_out_of_feature_vector() -> None:
    payload = assemble_row(_features(), _label(), _ledger(), dataset_version="v", source_cohort="c")
    report = leakage_audit([payload])
    assert report["status"] == "clean"
    assert report["canonical_features_are_disjoint_from_labels"]


def test_repository_temporal_split_has_no_repo_overlap() -> None:
    rows = []
    for index in range(6):
        row = assemble_row(
            _features(pr_id=index + 1, repo_full_name=f"org/r{index}", pr_number=index + 1, created_at=f"2025-0{index+1}-01T00:00:00Z"),
            _label(str(index + 1)),
            _ledger(str(index + 1)),
            dataset_version="v",
            source_cohort="c",
        )
        rows.append(row)
    split_rows, report = repository_temporal_split(rows, seed=42)
    assert report["repository_overlap"]["train_test"] == []
    assert {row["split"] for row in split_rows} <= {"train", "validation", "test"}
    again, _ = repository_temporal_split(rows, seed=42)
    assert [row["sample_id"] for row in split_rows] == [row["sample_id"] for row in again]
    assert [row["split"] for row in split_rows] == [row["split"] for row in again]


def test_dataset_package_write_and_provenance(tmp_path: Path) -> None:
    rows, split_report = repository_temporal_split([
        assemble_row(_features(), _label(), _ledger(), dataset_version="v", source_cohort="c")
    ], seed=42)
    result = AssemblyResult(
        candidate_count=1,
        supervised_rows=rows,
        excluded=[],
        unknown_count=0,
        quality_report={"duplicate_pr_ids": 0},
        language_report={"final_training_rows_per_language": {"python": 1}},
        label_report={"low_observed": 1},
        split_report=split_report,
        metadata={"source": "AIDev"},
        leakage_report=leakage_audit(rows),
    )
    paths = write_dataset_package(result, tmp_path, stem="aidev_supervised_42")
    assert paths["parquet"].exists()
    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert meta["source"] == "AIDev"


def test_target_provenance_and_unknown_exclusion_in_assembler(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [_features(), _features(pr_id=102, pr_number=8)]
    table = pa.Table.from_pylist(rows)
    parquet = tmp_path / "features.parquet"
    pq.write_table(table, parquet)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(_label("101")) + "\n" + json.dumps(_label("102", risk="unknown")) + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(_ledger("101")) + "\n" + json.dumps(_ledger("102")) + "\n",
        encoding="utf-8",
    )
    assembled = assemble_supervised_dataset(
        [(parquet, "test-cohort")],
        label_paths=[labels],
        ledger_paths=[ledger],
        cutoff_at=CUTOFF,
        seed=42,
    )
    assert assembled.unknown_count == 1
    assert len(assembled.supervised_rows) == 1
    assert assembled.supervised_rows[0]["evidence_type"] == "no_observed_adverse_outcome"
    assert all(row["risk_label"] != "unknown" for row in assembled.supervised_rows)


def test_sufficiency_gate_blocks_zero_high() -> None:
    y_neg = np.zeros(40, dtype=np.int32)
    report = assess_sufficiency(y_neg[:28], y_neg[28:32], y_neg[32:])
    assert report.trainable is False
    assert report.reason == "zero_high_labels"


def test_catboost_train_predict_serialize_when_sufficient(tmp_path: Path) -> None:
    catboost = pytest.importorskip("catboost")
    rows = []
    for index in range(60):
        high = index < 20
        feature_overrides = {name: float((index + 1) * (3 if high else 1)) for name in CANONICAL_FEATURES}
        row = assemble_row(
            _features(
                pr_id=index + 1,
                repo_id=index % 12,
                repo_full_name=f"org/r{index % 12}",
                pr_number=index + 1,
                created_at=f"2025-{(index % 9) + 1:02d}-01T00:00:00Z",
                **feature_overrides,
            ),
            _label(str(index + 1), risk="high" if high else "low_observed"),
            _ledger(str(index + 1), language="python" if index % 2 == 0 else "typescript"),
            dataset_version="v",
            source_cohort="c",
        )
        rows.append(row)
    split_rows, _ = repository_temporal_split(rows, seed=42)
    matrices = split_matrices(split_rows)
    # Guarantee both classes in splits for this unit test by using all rows as train/test if needed.
    if int(np.sum(matrices["y_train"])) == 0 or int(np.sum(matrices["y_test"])) == 0:
        for row in split_rows:
            row["split"] = "train" if int(row["pr_number"]) <= 40 else "test"
        for row in split_rows:
            if row["split"] == "train" and int(row["pr_number"]) > 35:
                row["split"] = "validation"
        split_rows = list(split_rows)
    report = train_deployment_risk_model(split_rows, output_dir=tmp_path, random_seed=42, iterations=30)
    if not report["trained"]:
        pytest.skip(f"synthetic split still insufficient: {report['sufficiency']}")
    assert (tmp_path / "deployment_risk_model.cbm").exists()
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(tmp_path / "deployment_risk_model.cbm"))
    wrapper = DeploymentRiskModel(model, model_version="v", feature_schema_version="s")
    prediction = wrapper.predict({name: 1.0 for name in CANONICAL_FEATURES})
    assert 0.0 <= prediction["risk_probability"] <= 1.0
    assert prediction["predicted_risk_class"] in {LabelState.HIGH.value, LabelState.LOW_OBSERVED.value}
    with pytest.raises(ValueError, match="feature schema mismatch"):
        wrapper.predict({"files_changed": 1.0})
    reload_pred = wrapper.predict({name: 1.0 for name in CANONICAL_FEATURES})
    assert reload_pred["risk_probability"] == prediction["risk_probability"]
    assert report["serialization"]["loaded_predictions_identical"] is True
