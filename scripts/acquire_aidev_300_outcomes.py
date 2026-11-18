"""Build Phase 7G evidence and labels for the immutable existing AIDev 300."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.phase7c_contract import LabelState
from dataset.phase7d_labeler import reconcile_label
from dataset.phase7d_outcome_evidence import GitHubOutcomeEvidenceProvider, OutcomeEvidenceResult, UnavailableOutcomeEvidenceProvider
from dataset.phase7f_identity_acquisition import GitHubClient


PREVIOUS_401_PREFIX = "outcome_acquisition_failed: github_http_401: Bad credentials"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else None


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _label_row(row: dict[str, Any], result: OutcomeEvidenceResult, cutoff_at: datetime) -> dict[str, Any]:
    merged_at = _parse_time(row.get("merged_at"))
    resolved = row.get("acquisition_status") == "resolved"
    label = reconcile_label(merged_at=merged_at, audited_at=cutoff_at, result=result)
    failed = resolved and merged_at is not None and not result.evaluated
    return {
        "sample_id": row["sample_id"], "repository": row["repo_full_name"], "pr_number": row["pr_number"],
        "resolved": resolved, "merged": merged_at is not None,
        "mature": bool(merged_at and merged_at + timedelta(days=90) <= cutoff_at),
        "risk_label": label.label.value, "confidence": label.confidence, "evidence_type": label.evidence_type,
        "evidence_references": list(label.references), "observation_start": label.observation_start.isoformat() if label.observation_start else None,
        "observation_end": label.observation_end.isoformat() if label.observation_end else None,
        "label_policy_version": label.policy_version, "unknown_or_conflicted_reason": label.unavailable_reason,
        "evidence_acquisition_failure": failed,
    }


def _evidence_rows(sample_id: str, result: OutcomeEvidenceResult) -> list[dict[str, Any]]:
    return [{key: (value.value if key == "classification" else value.isoformat() if isinstance(value, datetime) else value)
             for key, value in item.__dict__.items()} | {"sample_id": sample_id}
            for item in result.evidence]


def _report(label_rows: list[dict[str, Any]], cutoff_at: datetime, output_dir: Path) -> dict[str, Any]:
    counts = Counter(row["risk_label"] for row in label_rows)
    return {"total": 300, "resolved": sum(row["resolved"] for row in label_rows), "merged": sum(row["merged"] for row in label_rows),
            "mature": sum(row["mature"] for row in label_rows), "high": counts[LabelState.HIGH.value], "medium": counts[LabelState.MEDIUM.value],
            "low_observed": counts[LabelState.LOW_OBSERVED.value], "unknown": counts[LabelState.UNKNOWN.value],
            "conflicted": counts[LabelState.CONFLICTED.value], "evidence_acquisition_failures": sum(row["evidence_acquisition_failure"] for row in label_rows),
            "labelable_binary_count": counts[LabelState.HIGH.value] + counts[LabelState.LOW_OBSERVED.value],
            "cutoff_at": cutoff_at.isoformat().replace("+00:00", "Z"),
            "output_paths": [str(output_dir / name) for name in ("phase7d_labels_300.jsonl", "phase7d_outcome_evidence_300.jsonl", "phase7d_outcome_report.json")]}


def build_artifacts(*, ledger_path: Path, output_dir: Path, cutoff_at: datetime, client: GitHubClient | None) -> dict[str, Any]:
    """Label all fixed rows; only mature, merged, resolved rows receive scans."""
    rows = _load_ledger(ledger_path)
    if len(rows) != 300:
        raise ValueError(f"identity ledger has {len(rows)} rows; expected fixed 300")
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = (GitHubOutcomeEvidenceProvider(client, output_dir / "evidence_cache") if client else
                UnavailableOutcomeEvidenceProvider("github_token_missing_offline_acquisition"))
    label_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    failures = 0
    for row in rows:
        merged_at = _parse_time(row.get("merged_at"))
        resolved = row.get("acquisition_status") == "resolved"
        if resolved and merged_at is not None:
            result = provider.collect(repository=str(row["repo_full_name"]), pr_id=str(row["pr_number"]),
                                      merge_sha=row.get("merge_commit_sha"), merge_at=merged_at,
                                      original_head_sha=row.get("final_pr_head_sha"),
                                      original_paths=(str(item["path"]) for item in row.get("changed_files", []) if item.get("path")))
        else:
            result = OutcomeEvidenceResult(None, (), False, str(row.get("failure_reason") or "unmerged"))
        label_row = _label_row(row, result, cutoff_at)
        failures += label_row["evidence_acquisition_failure"]
        label_rows.append(label_row)
        evidence_rows.extend(_evidence_rows(row["sample_id"], result))
    labels_path = output_dir / "phase7d_labels_300.jsonl"
    evidence_path = output_dir / "phase7d_outcome_evidence_300.jsonl"
    report_path = output_dir / "phase7d_outcome_report.json"
    for path, values in ((labels_path, label_rows), (evidence_path, evidence_rows)):
        path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")
    counts = Counter(row["risk_label"] for row in label_rows)
    report = {"total": 300, "resolved": sum(row["resolved"] for row in label_rows), "merged": sum(row["merged"] for row in label_rows),
              "mature": sum(row["mature"] for row in label_rows), "high": counts[LabelState.HIGH.value], "medium": counts[LabelState.MEDIUM.value],
              "low_observed": counts[LabelState.LOW_OBSERVED.value], "unknown": counts[LabelState.UNKNOWN.value],
              "conflicted": counts[LabelState.CONFLICTED.value], "evidence_acquisition_failures": failures,
              "labelable_binary_count": counts[LabelState.HIGH.value] + counts[LabelState.LOW_OBSERVED.value],
              "cutoff_at": cutoff_at.isoformat().replace("+00:00", "Z"),
              "output_paths": [str(labels_path), str(evidence_path), str(report_path)]}
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return report


def retry_previous_401(*, ledger_path: Path, output_dir: Path, cutoff_at: datetime, client: GitHubClient) -> dict[str, Any]:
    """Retry only persisted Phase 7D 401 snapshots and patch just their rows."""
    ledger_rows = _load_ledger(ledger_path)
    labels_path = output_dir / "phase7d_labels_300.jsonl"
    evidence_path = output_dir / "phase7d_outcome_evidence_300.jsonl"
    report_path = output_dir / "phase7d_outcome_report.json"
    label_rows = _load_jsonl(labels_path)
    evidence_rows = _load_jsonl(evidence_path)
    if len(ledger_rows) != 300 or len(label_rows) != 300:
        raise ValueError("targeted retry requires the existing immutable 300-row ledger and 300-row labels")
    target_ids = {str(row["sample_id"]) for row in label_rows
                  if str(row.get("unknown_or_conflicted_reason") or "").startswith(PREVIOUS_401_PREFIX)}
    if not target_ids:
        raise ValueError("no previous github_http_401 outcome snapshots found")
    ledger_by_id = {str(row["sample_id"]): row for row in ledger_rows}
    if target_ids - set(ledger_by_id.keys()):
        raise ValueError("a targeted retry label is absent from the immutable identity ledger")
    provider = GitHubOutcomeEvidenceProvider(client, output_dir / "evidence_cache")
    replacements: dict[str, dict[str, Any]] = {}
    replacement_evidence: dict[str, list[dict[str, Any]]] = {}
    statuses: list[dict[str, Any]] = []
    for sample_id in (str(row["sample_id"]) for row in label_rows if str(row["sample_id"]) in target_ids):
        row = ledger_by_id[sample_id]
        merged_at = _parse_time(row.get("merged_at"))
        if row.get("acquisition_status") != "resolved" or merged_at is None:
            raise ValueError(f"401 retry target is not a resolved merged identity: {sample_id}")
        result = provider.collect(repository=str(row["repo_full_name"]), pr_id=str(row["pr_number"]),
                                  merge_sha=row.get("merge_commit_sha"), merge_at=merged_at,
                                  original_head_sha=row.get("final_pr_head_sha"),
                                  original_paths=(str(item["path"]) for item in row.get("changed_files", []) if item.get("path")),
                                  retry_previous_401=True)
        replacement = _label_row(row, result, cutoff_at)
        replacements[sample_id] = replacement
        replacement_evidence[sample_id] = _evidence_rows(sample_id, result)
        statuses.append({"sample_id": sample_id, "repository": row["repo_full_name"], "pr_number": row["pr_number"],
                         "acquisition_status": "evaluated" if result.evaluated else "unavailable",
                         "reason": result.unavailable_reason})
    updated_labels = [replacements.get(str(row["sample_id"]), row) for row in label_rows]
    updated_evidence = [row for row in evidence_rows if str(row.get("sample_id")) not in target_ids]
    for row in label_rows:
        updated_evidence.extend(replacement_evidence.get(str(row["sample_id"]), ()))
    for path, values in ((labels_path, updated_labels), (evidence_path, updated_evidence)):
        path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")
    report = _report(updated_labels, cutoff_at, output_dir)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    report["targeted_retry"] = statuses
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire post-merge evidence for the existing AIDev 300 only.")
    parser.add_argument("--ledger", type=Path, default=Path("data/processed/aidev_300_acquisition/identity_ledger.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/aidev_300_acquisition/outcomes"))
    parser.add_argument("--cutoff-at", default="2026-09-06T00:00:00Z")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Write explicit UNKNOWN failures without contacting GitHub.")
    parser.add_argument("--retry-previous-401", action="store_true", help="Retry only persisted Phase 7D github_http_401 snapshots.")
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token and not (args.allow_unauthenticated or args.offline):
        raise SystemExit("GITHUB_TOKEN or GH_TOKEN is required (or deliberately pass --allow-unauthenticated).")
    cutoff = _parse_time(args.cutoff_at)
    if cutoff is None:
        raise SystemExit("--cutoff-at must be a timezone-aware ISO-8601 timestamp")
    if args.retry_previous_401 and (args.offline or args.allow_unauthenticated):
        raise SystemExit("--retry-previous-401 requires authenticated GitHub acquisition")
    client = None if args.offline else GitHubClient(token, args.output_dir.parent / "raw_github")
    report = (retry_previous_401(ledger_path=args.ledger, output_dir=args.output_dir, cutoff_at=cutoff, client=client)
              if args.retry_previous_401 else build_artifacts(ledger_path=args.ledger, output_dir=args.output_dir, cutoff_at=cutoff, client=client))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
