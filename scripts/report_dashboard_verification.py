"""Render an honest audit from saved live runs, without generating evidence."""
import json
from pathlib import Path


def main():
    reports = Path('reports')
    rows = json.loads((reports / 'dashboard_verified.json').read_text())
    inventory = json.loads((reports / 'dashboard_inventory_before.json').read_text())
    lines = [
        '# Dashboard repair verification — incomplete', '',
        'These are real provider runs against acquired GitHub commits, persisted in the application database. '
        'A COMPLETE analysis row does not mean a validated repair. No fresh selectable candidate has been established.', '',
        '## A. Root causes and changes', '',
        '- `backend/app/services/workflow.py:generate_candidate`: Groq-specific early exits prevented bounded semantic retries; repair and test generation were coupled; original regression failures skipped original integration execution. These now use the generic three-attempt loop, separate generation, and independent baseline checks.',
        '- `backend/app/services/repair.py:ConfiguredProvider.propose_repair`: malformed JSON and incomplete diffs were returned by the configured provider. Requests now include distinct base/head evidence and compact source context; malformed output is rejected with retry feedback. Missing headers and hunk ranges are never synthesized.',
        '- `backend/app/services/test_generation.py:_ruff_preflight`: generated tests were being automatically rewritten. Preflight now reports errors without rewriting the provider output; integration may be absent when no real path exists.',
        '- `backend/app/services/validation.py:validate` and `native_build.py`: repository imports, explicit test/test.py discovery, and native-extension build prerequisites were incomplete. Validation now uses checkout import paths, discovers that test layout, and can build repository-declared native extensions with the installed Windows toolchain.',
        '- `backend/app/services/analysis.py:_acquire` and `workflow.py:generate_candidate`: explicit base/head snapshots now remain distinct during candidate reanalysis. Candidate source mutation during validation blocks selection.',
        '- `backend/app/services/ci_tools.py:snyk`: unbound host scan evidence no longer becomes a candidate PASS. Missing scanners remain unavailable.',
        '- `frontend/src/main.tsx`: unavailable scanner copy no longer asserts local success; candidate measured reanalysis and optional configured Sonar results are visible.', '',
        '## B. Changed files', '',
        'Platform: `backend/app/services/{workflow,repair,test_generation,validation,native_build,candidate,analysis,ci_tools}.py`; `frontend/src/main.tsx`.', '',
        'Checks and utilities: `tests/test_repository_agnostic_repair.py`, `tests/test_final_architecture_completion.py`, `tests/test_ci_tooling.py`, `tests/test_live_repair_completion.py`, `tests/test_provider_response_robustness.py`; `scripts/verify_dashboard_live.py`, `scripts/acquire_dashboard_sources.py`, `scripts/verify_acquired_dashboard.py`, `scripts/report_dashboard_verification.py`, `scripts/salvage_persisted_candidate.py`.', '',
        '## C. Local checks', '',
    ]
    for name in ('full_tests.log', 'frontend_lint.log', 'frontend_build.log', 'frozen_demo_verification.log'):
        log = reports / name
        lines += [f'[{name}]({name})', '```text', '\n'.join(log.read_text(errors='replace').splitlines()[-5:]) if log.exists() else 'Not recorded', '```', '']
    lines += ['The preserved frozen demo is historical evidence, not a fresh provider success. The updated HTTP backend passed `/api/health` on 2026-09-29.', '',
              '## D. Per-PR results', '',
              '| Repository / PR | Diff bytes | Analysis | Graph nodes/edges | Impact score | Patch | Regression gen/pass | Integration gen/pass | Candidate validation | Reanalysis | Selectable |',
              '|---|---:|---|---|---:|---|---|---|---|---|---|']
    for row in sorted(rows, key=lambda x: (x['repository'], x['number'])):
        r = row['result']; a = r.get('analysis', {}); p = r.get('repair', {}); c = p.get('candidate') or {}
        graph = a.get('graph') or {}; gen = r.get('test_generation') or {}; checks = (c.get('validation') or {}).get('checks') or {}
        diff = Path('.tmp/dashboard-sources') / (row['pr_id'] + '.diff')
        def test_result(kind, check):
            return f"{len(gen.get(kind) or [])} / {checks.get(check, {}).get('status', 'not run')}"
        lines += [f"| {row['repository']} #{row['number']} | {diff.stat().st_size if diff.exists() else 'unknown'} | {'complete' if a.get('analysis_complete') else 'incomplete'} | {len(graph.get('nodes', []))}/{len(graph.get('edges', []))} | {a.get('impact_summary', {}).get('impact_score', 'unknown')} | {c.get('status', 'not created')} | {test_result('generated_tests', 'regression')} | {test_result('generated_integration_tests', 'targeted_integration')} | {(c.get('validation') or {}).get('status', 'not run')} | {(c.get('reanalysis') or {}).get('status', 'not run')} | {bool(c.get('selectable'))} |"]
    lines += ['', '## E. Persisted evidence and generated artifacts', '',
              '[Full original inventory, including IDs, titles, branches, SHAs and initial analysis](dashboard_inventory_before.json).', '',
              '[Initial live reproduction](dashboard_live_before_fixes.json). [Latest per-PR provider evidence and presentation](dashboard_verified.json).', '',
              'No validated generated patch/test excerpts are presented as successful: the fresh runs below have not reached that state.', '']
    for row in rows:
        r = row['result']; p = r.get('repair') or {}
        lines += [f"### {row['repository']} #{row['number']}", '', f"Internal PR ID: `{row['pr_id']}`. Persisted analysis ID: `{row['analysis_id']}`.", '', str(p.get('issue_summary') or ''), '']
        for attempt in r.get('repair_attempts', []):
            lines += [f"- {attempt.get('status')}: {attempt.get('failure_reason') or 'See persisted evidence.'}"]
        if p.get('provider_error'):
            lines += ['', '```json', json.dumps(p['provider_error'], indent=2), '```']
        if p.get('patch'):
            lines += ['', 'Rejected provider patch excerpt (not applied or approved):', '```diff', '\n'.join(p['patch'].splitlines()[:24]), '```']
        lines += ['']
    lines += ['## F. Remaining limitations', '',
              '- Calculator #2 has identical base/head trees. Its branch correction has not been made. The most recent installation check still returned Contents=read and Pull Requests=read, despite the planned permission update.',
              '- The configured Groq model has repeatedly returned bare hunk headers or malformed structured output; rate/context limits also occurred. Strict rejection remains active. Provider failures are not platform success.',
              '- Optional remote provider code execution is disabled. Automatic approval review rejected activation because it introduces remote execution and an additional sandbox destination for source; separate user approval is pending.',
              '- Dependency scans require working scanner configuration. No missing scan is reported as passed.',
              '- Mocked provider boundary tests prove control flow, not the quality of live model outputs. Fresh end-to-end success across all dashboard PRs remains unverified.', '']
    (reports / 'dashboard_verification.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
