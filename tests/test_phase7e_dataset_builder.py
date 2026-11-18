from datetime import UTC, datetime

from dataset.phase7c_contract import DatasetManifest
from dataset.phase7e_dataset_builder import build_candidate_dataset
from tests.test_phase7c_candidate_audit import candidate


def test_builder_keeps_features_empty_until_controlled_extraction() -> None:
    manifest = DatasetManifest("v1", "phase7c-42-v1", "pinned", "policy", {}, {}, {}, (), 90, 90,
                               datetime(2026, 1, 1, tzinfo=UTC), candidate_count=100, selected_count=0)
    built = build_candidate_dataset([candidate(str(i), network=f"n{i}") for i in range(100)], requested_rows=100, manifest=manifest)
    assert len(built.samples) == 100
    assert built.features_42 == ()
    assert built.attrition_report.balancing_performed is False
