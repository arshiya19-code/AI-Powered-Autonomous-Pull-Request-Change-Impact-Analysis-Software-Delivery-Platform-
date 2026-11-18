from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from backend.app import database, product, github
from backend.app.presentation import build_presentation
from backend.app.services.github import GitHubIntegrationError, GitHubService
from backend.app.services.stages import transition
from backend.app.services.candidate import CandidateWorkspaceService
from backend.app.services.test_generation import TestGenerationService
from backend.app.services.workflow import DeliveryWorkflowService
from backend.app.schemas.models import RepairResult, TestGenerationResult
from tests.test_github_integration import github_clients, install, pr_payload
from tests.test_ai_repair_validation_loop import make_repo, workflow, service, LoopProvider


def test_background_acquisition_failure_persists_stage_and_skips_graph(github_clients, monkeypatch, caplog):
    owner, _, factory, _ = github_clients
    connection = install(owner, factory, monkeypatch)
    with factory() as db:
        pr = database.PullRequest(repository_connection_id=connection.id, **github.service.pull_request_values(pr_payload()))
        db.add(pr); db.commit(); pr_id = pr.id
    monkeypatch.setattr(github.service, 'get_pull_request', lambda *a: pr_payload())
    def fail(*a): raise GitHubIntegrationError('No file differences between actual base and head', 'diff_unavailable')
    monkeypatch.setattr(github.service, 'materialize_pull_request', fail)
    monkeypatch.setattr(product, 'executor', SimpleNamespace(submit=lambda fn, *a: fn(*a)))
    response = owner.post(f'/api/github/pull-requests/{pr_id}/analyze')
    detail = owner.get('/api/product/analyses/' + response.json()['id']).json()
    assert detail['status'] == 'FAILED'
    with factory() as db:
        result = db.get(database.Analysis, detail['id']).result
    assert result['stages']['PR_ACQUIRED']['status'] == 'FAILED'
    for key in ('CODE_PARSED', 'GRAPH_BUILT', 'IMPACT', 'RISK'):
        assert result['stages'][key]['status'] == 'SKIPPED'
    assert 'graph' not in result.get('analysis', {})
    assert 'Traceback' in caplog.text
    assert result['error']['message'].startswith('No file differences')
    assert any(x['status'] == 'FAILED' for x in detail['presentation']['trace'])


@pytest.mark.parametrize('failed_event,expected', [('PR_ACQUISITION_STARTED','PR_ACQUIRED'), ('GRAPH_STARTED','GRAPH_BUILT'), ('RISK_STARTED','RISK')])
def test_first_running_stage_is_failed(failed_event, expected):
    stages = transition({}, failed_event)
    stages = transition(stages, 'WORKFLOW_FAILED', {'message':'actual error', 'type':'ValueError'})
    assert stages[expected]['status'] == 'FAILED'
    assert stages[expected]['duration'] >= 0
    assert all(s['status'] not in ('PENDING', 'RUNNING') for s in stages.values())


def test_stacked_acquisition_fetches_exact_shas_and_empty_diff_is_explicit(tmp_path, monkeypatch):
    calls=[]
    def git(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(subprocess, 'run', git)
    svc=GitHubService(); monkeypatch.setattr(svc,'installation_token',lambda *a:'test-token')
    diff, sources, evidence = svc.materialize_pull_request('1','owner/repo','a'*40,'b'*40,tmp_path)
    assert diff == '' and sources == {}
    assert evidence[-1]['type'] == 'SOURCE_SNAPSHOT_ACQUIRED'
    assert ['git','fetch','--quiet','--no-tags','--depth=1','origin','a'*40] in calls
    assert ['git','fetch','--quiet','--no-tags','--depth=1','origin','b'*40] in calls
    assert ['git','diff','--no-ext-diff','--binary','a'*40,'b'*40] in calls


def test_production_patch_rejection_never_starts_validation(tmp_path):
    make_repo(tmp_path)
    provider=LoopProvider()
    provider.propose_repair=lambda ctx: RepairResult(status='available',patch='not a unified diff')
    result=service(provider).generate_candidate(workflow(),root=str(tmp_path))
    assert all(a['status']=='PATCH_REJECTED' for a in result.repair_attempts)
    assert all('CANDIDATE_VALIDATION_STARTED' not in a['state_history'] for a in result.repair_attempts)


def test_bad_test_patch_preserves_valid_candidate_source_without_validation(tmp_path):
    make_repo(tmp_path)
    provider=LoopProvider(); original=provider.propose_repair
    provider.propose_repair=lambda ctx: RepairResult(status='available',patch='invalid') if ctx.get('purpose') else original(ctx)
    result=service(provider).generate_candidate(workflow(),root=str(tmp_path))
    assert result.repair.candidate['files'][0]['original_code'] != result.repair.candidate['files'][0]['candidate_code']
    assert all(a['status']=='TEST_GENERATION_FAILED' for a in result.repair_attempts)
    assert all('CANDIDATE_VALIDATION_STARTED' not in a['state_history'] for a in result.repair_attempts)
    assert result.repair.candidate['selectable'] is False
    assert result.repair.candidate['status'] == 'rejected'
    assert result.repair.candidate['original_preserved'] is True
    assert (tmp_path/'app.py').read_text() == 'def add(a, b):\n    return a * b\n'
    assert len(provider.repair_calls) == 1
    assert 'candidate_error' in result.repair_attempts[0]['failure_feedback']


def test_generated_source_cannot_bypass_test_only_patch_gate(tmp_path):
    make_repo(tmp_path); svc=CandidateWorkspaceService(); candidate=svc.create(tmp_path)
    try:
        with pytest.raises(ValueError,match='production source'):
            svc.add_generated_tests(candidate, TestGenerationResult(generated_tests=[{'path':'app.py','content':'def test_bad():\n    assert True\n','type':'regression','purpose':'malicious overwrite'}]))
    finally: candidate.cleanup()
    assert 'return a * b' in (tmp_path/'app.py').read_text()


def test_retry_of_persisted_valid_patch_only_calls_test_generation(tmp_path):
    make_repo(tmp_path)
    provider = LoopProvider()
    original = provider.propose_repair
    provider.propose_repair = lambda ctx: RepairResult(status='available', patch='invalid') if ctx.get('purpose') else original(ctx)
    worker = service(provider)
    first = worker.generate_candidate(workflow(), root=str(tmp_path))
    patch = first.repair.candidate['patch']
    assert len(provider.repair_calls) == 1
    provider.propose_repair = original
    second = worker.generate_candidate(first, root=str(tmp_path))
    assert len(provider.repair_calls) == 1
    assert second.repair.candidate['patch'] == patch
    assert second.repair.candidate['selectable']


def test_generated_test_contract_rejects_unified_patch_payload():
    with pytest.raises(ValueError, match='tests array'):
        TestGenerationService._parse_response({'patch':'--- a/tests/test_app.py\n+++ b/tests/test_app.py\n'})


def test_source_context_reaches_provider_and_reanalysis_uses_all_sources(tmp_path):
    make_repo(tmp_path); (tmp_path/'caller.py').write_text('from app import add\n')
    provider=LoopProvider(); svc=service(provider)
    requests=[]; original=svc.analysis.analyze
    svc.analysis.analyze=lambda request: (requests.append(request),original(request))[1]
    result=svc.generate_candidate(workflow(),root=str(tmp_path))
    assert 'caller.py' in provider.repair_calls[0]['source_context']
    assert 'caller.py' in requests[0].sources
    assert Path(requests[0].repository_path) != tmp_path
    assert result.repair.candidate['files'][0]['added_lines']==1
    assert result.repair.candidate['files'][0]['removed_lines']==1


def test_candidate_reanalysis_failure_disables_selection(tmp_path):
    make_repo(tmp_path); svc=service(LoopProvider())
    def fail(request): raise ValueError('graph failure')
    svc.analysis.analyze=fail
    result=svc.generate_candidate(workflow(),root=str(tmp_path))
    assert result.repair.candidate['selectable'] is False
