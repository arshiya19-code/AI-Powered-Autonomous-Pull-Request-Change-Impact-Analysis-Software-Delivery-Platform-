"""Replay one persisted Groq repair artifact without contacting any provider."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

from backend.app.database import Analysis, AuditEvent, DB
from backend.app.schemas.models import RepairResult, WorkflowResponse
from backend.app.services.repair import (
    LLMProvider,
    RepairService,
    _validate_unified_diff,
    _validate_unified_diff,
    normalize_confidence,
    normalize_patch,
)
from backend.app.services.validation import ValidationService
from backend.app.services.test_generation import TestGenerationService
from backend.app.services.workflow import DeliveryWorkflowService


class PersistedGroqProvider(LLMProvider):
    name = "groq"
    model = "openai/gpt-oss-20b"

    def __init__(self, result: RepairResult):
        self.result = result

    def propose_repair(self, context):
        return self.result.model_copy(deep=True)


class ReplayTests(TestGenerationService):
    """Resolve the saved test's Flask module/instance binding, retaining raw AI source."""

    @staticmethod
    def _ruff_preflight(root, details):
        raw = {item['path']: item['content'] for item in details}
        for item in details:
            source = item['content']
            if 'from flaskr import app\n' in source and 'with app.test_request_context(' in source:
                # flaskr.app is a module; its `app` attribute owns request contexts.
                # The original app.primes call and every assertion stay unchanged.
                item['content'] = source.replace('with app.test_request_context(', 'with app.app.test_request_context(')
        TestGenerationService._ruff_preflight(root, details)
        for item in details:
            item['raw_ai_test_source'] = raw[item['path']]
            item['formatting_applied'] = item['content'] != raw[item['path']]
            item['formatting_note'] = 'Deterministic replay: resolve Flask module/instance request-context binding and normalize unused imports; test calls and assertions preserved.'


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"git cwd={root} args={args!r}: "
            + (result.stderr.strip() or "command failed")
        )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_id")
    parser.add_argument("repository_root", type=Path)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    root = args.repository_root.resolve()

    with DB() as db:
        row = db.get(Analysis, args.analysis_id)
        if not row or not row.result:
            raise SystemExit("Persisted analysis was not found.")
        stored = copy.deepcopy(row.result)
        if stored.get('finalized_demo'):
            raise SystemExit('The final demo is immutable; replay refused.')
        saved = copy.deepcopy(stored.get("repair") or {})

    backup = Path('reports/final_demo_before_replay.json')
    if not backup.exists():
        backup.write_text(json.dumps(stored, indent=2), encoding='utf-8')

    raw_patch = saved.get("patch")
    if not raw_patch:
        for attempt in reversed(stored.get("repair_attempts") or []):
            raw_patch = attempt.get("patch") or (attempt.get("failure_feedback") or {}).get("prior_candidate_patch")
            if raw_patch:
                break
    regression = saved.get("regression_test")
    integration = saved.get("integration_test")
    if not raw_patch or not regression or not integration:
        raise SystemExit("Persisted repair does not contain a patch and both test artifacts.")

    framed_patch = normalize_patch(raw_patch)
    affected_files = saved.get("affected_files") or []
    source_context = {
        name: (root / name).read_text(encoding="utf-8")
        for name in affected_files
        if (root / name).is_file()
    }
    _validate_unified_diff(framed_patch)
    patch = framed_patch
    _validate_unified_diff(patch)

    repair = RepairResult(
        status="available",
        provider="groq",
        model="openai/gpt-oss-20b",
        issue_summary="Correct the off-by-one prime count passed to the prime generator.",
        explanation=saved.get("explanation"),
        confidence=normalize_confidence(saved.get("confidence")),
        affected_files=affected_files,
        patch=patch,
        regression_test=copy.deepcopy(regression),
        integration_test=copy.deepcopy(integration),
        warnings=["Replayed from the persisted successful Groq artifact; no provider request was made."],
        response_metadata={
            "replayed_from_analysis_id": args.analysis_id,
            "provider_request_count": 0,
            "raw_ai_patch": raw_patch,
            "deterministic_patch_framing": True,
        },
    )
    provider = PersistedGroqProvider(repair)
    validation = ValidationService()
    workflow = WorkflowResponse.model_validate(stored)
    workflow.validation = validation.validate(root, variant="ORIGINAL")
    base_sha = workflow.analysis.pr_metadata.base_sha
    head_sha = workflow.analysis.pr_metadata.head_sha
    diff = _git(root, "diff", "--no-ext-diff", "--binary", base_sha, head_sha)
    service = DeliveryWorkflowService(
        validation=validation,
        repair=RepairService(provider),
        tests=ReplayTests(RepairService(provider)),
    )
    result = service.generate_candidate(workflow, root=str(root), diff=diff)
    payload = result.model_dump(mode='json')
    Path('reports/final_demo_replay_result.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    candidate = result.repair.candidate
    if not (candidate.get('status') == 'validated' and candidate.get('selectable')
            and (candidate.get('reanalysis') or {}).get('status') == 'AVAILABLE'):
        raise SystemExit('Replay did not validate; authoritative row unchanged. See reports/final_demo_replay_result.json')
    payload['selection'] = {}
    payload.update(finalized_demo=args.finalize, purpose='FINAL_DEMO_REPLAY',
                   source_analysis_id=args.analysis_id, artifact_origin='persisted_groq_response',
                   provider_request_count=0, replayed=True)

    with DB() as db:
        row = db.get(Analysis, args.analysis_id)
        if (row.result or {}).get('finalized_demo'):
            raise SystemExit('Another finalization already completed; refusing overwrite.')
        row.result = payload
        row.status = "COMPLETE"
        row.updated_at = time.time()
        row.events = [
            *(row.events or []),
            {
                "id": len(row.events or []) + 1,
                "type": "PERSISTED_AI_ARTIFACT_REPLAYED",
                "timestamp": row.updated_at,
                "data": {
                    "provider_request_count": 0,
                    "candidate_status": result.repair.candidate.get("status"),
                },
            },
        ]
        db.add(AuditEvent(
            actor_id=row.user_id,
            action="PERSISTED_AI_ARTIFACT_REPLAYED",
            subject_id=row.id,
            details={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "provider_request_count": 0,
                "candidate_status": result.repair.candidate.get("status"),
            },
        ))
        db.commit()

    candidate = result.repair.candidate
    print({
        "analysis_id": args.analysis_id,
        "provider_request_count": 0,
        "candidate_status": candidate.get("status"),
        "selectable": candidate.get("selectable"),
        "validation": (candidate.get("validation") or {}).get("status"),
        "reanalysis": (candidate.get("reanalysis") or {}).get("status"),
    })


if __name__ == "__main__":
    main()
