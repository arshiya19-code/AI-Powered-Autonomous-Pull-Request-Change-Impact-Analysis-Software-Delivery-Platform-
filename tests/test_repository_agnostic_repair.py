import pytest
from backend.app.schemas.models import RepairResult
from backend.app.services.repair import ProviderResponseError, _compact_repair_context
from backend.app.services.test_generation import TestGenerationService
from backend.app.services.validation import ValidationService
from tests.test_ai_repair_validation_loop import LoopProvider, make_repo, service, workflow

@pytest.mark.parametrize('failure', ['patch', 'structure', 'validation'])
def test_groq_uses_bounded_semantic_retries(tmp_path, failure):
    make_repo(tmp_path)
    class Provider(LoopProvider):
        name = 'groq'
        def propose_repair(self, context):
            if context.get('purpose') == 'regression_test_generation':
                return super().propose_repair(context)
            if context['attempt'] == 1 and failure == 'structure':
                self.repair_calls.append(context)
                raise ProviderResponseError('integration_test.source is missing')
            value = super().propose_repair(context)
            value.provider = self.name
            if context['attempt'] == 1 and failure == 'patch':
                value.patch = 'diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@\n-no\n+yes\n'
            return value
    provider = Provider()
    result = service(provider).generate_candidate(workflow(), root=str(tmp_path), diff='real diff')
    if failure == 'validation':
        assert len(provider.repair_calls) == 1
        assert result.repair_attempts[-1]['production_patch_frozen']
    else:
        assert 2 <= len(provider.repair_calls) <= 3
        assert provider.repair_calls[1]['failure_feedback']
    assert result.repair.candidate['selectable']


def test_context_keeps_base_head_and_nested_evidence_distinct():
    compact = _compact_repair_context({'repository':'owner/repo','base_sha':'a','head_sha':'b','base_source_context':{'pkg.py':'old'},'source_context':{'pkg.py':'new'},'shap':{'shap_status':'available','top_risk_factors':[{'feature':'lines_added','value':1}]},'repository_layout':['pkg.py']})
    assert compact['base_source_snippets']['pkg.py'] == 'old'
    assert compact['relevant_original_source_snippets']['pkg.py'] == 'new'
    assert compact['base_sha'] != compact['head_sha']
    assert compact['shap']


def test_compacting_context_never_changes_source_or_retry_line_numbers():
    import copy

    source = ''.join(f'# Original source line {number}\n' for number in range(1, 160))
    context = {'source_context': {'module.py': source},
               'changed_symbols': [{'file': 'module.py', 'name': 'changed', 'start_line': 120, 'end_line': 124}]}
    original = copy.deepcopy(context)
    first = _compact_repair_context(context)
    second = _compact_repair_context(context)
    assert context == original
    assert first == second
    assert '120 | # Original source line 120' in first['numbered_head_source_windows']['module.py']
    assert first['head_source_window_ranges']['module.py'] == {'first_line': 118, 'last_line': 126}


def test_single_module_can_have_no_integration(tmp_path):
    (tmp_path/'app.py').write_text('def value():\n    return 3\n')
    repair=RepairResult(status='available',regression_test={'path':'tests/ai_generated/test_value.py','target':'app.value','reason':'Protect value','source':'from app import value\n\n\ndef test_value():\n    assert value() == 3\n'})
    result=TestGenerationService().from_repair(repair,root=tmp_path)
    assert result.generated_tests and not result.generated_integration_tests


@pytest.mark.parametrize('layout', ['module','src'])
@pytest.mark.parametrize('test_path', ['tests/test_value.py','test/test_value.py','checks/value_test.py'])
def test_repository_layout_discovery_executes_real_tests(tmp_path,layout,test_path):
    source=tmp_path/('src/example/__init__.py' if layout=='src' else 'example.py')
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text('def value():\n    return 3\n')
    target=tmp_path/test_path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('from example import value\n\ndef test_value():\n    assert value() == 3\n')
    result=ValidationService().validate(tmp_path,requested=['build','unit'])
    assert result.checks['unit']['status']=='PASSED',result.checks


def test_configured_provider_accepts_patch_only_contract(monkeypatch):
    import json
    from backend.app.services.repair import ConfiguredProvider
    from tests.test_provider_response_robustness import VALID, Response
    value = {key: VALID[key] for key in ("explanation", "confidence", "affected_files", "patch")}
    envelope = json.dumps({"choices": [{"message": {"content": json.dumps(value)}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(envelope))
    result = ConfiguredProvider("groq", "configured-model", "https://provider.test", "test-key").propose_repair({})
    assert result.patch == value["patch"]
    assert not result.regression_test

@pytest.mark.parametrize('enabled', [False, True])
def test_provider_code_execution_requires_explicit_opt_in(monkeypatch, enabled):
    import json
    from backend.app.services.repair import ConfiguredProvider
    from tests.test_provider_response_robustness import VALID, Response
    captured = {}
    value = {key: VALID[key] for key in ('explanation', 'confidence', 'affected_files', 'patch')}
    def request(req, **kwargs):
        captured.update(json.loads(req.data))
        return Response(json.dumps({'choices': [{'message': {'content': json.dumps(value)}}]}).encode())
    monkeypatch.setattr('urllib.request.urlopen', request)
    monkeypatch.setenv('LLM_PATCH_CODE_TOOL', '1' if enabled else '0')
    ConfiguredProvider('groq', 'openai/gpt-oss-20b', 'https://provider.test', 'test-key').propose_repair({'failure_feedback': {'patch_rejection_reason': 'Bare hunk rejected'}})
    assert ('tools' in captured) is enabled
    assert ('response_format' in captured) is not enabled


def test_provider_retry_handles_milliseconds_without_busy_retry():
    import json
    from backend.app.services.repair import _retry_after_seconds
    raw=json.dumps({'error': {'message': 'Please try again in 7.5ms.'}}).encode()
    assert _retry_after_seconds({}, raw) >= 1
    assert _retry_after_seconds({'Retry-After':'0'}, raw) >= 1


def test_candidate_source_mutation_never_becomes_selectable(tmp_path):
    from tests.test_ai_repair_validation_loop import LoopValidation
    make_repo(tmp_path)
    provider=LoopProvider()
    worker=service(provider)
    class MutatingValidation(LoopValidation):
        def validate(self, root, generated_test_files=None):
            value=super().validate(root,generated_test_files)
            (root/'app.py').write_text('def add(a, b):\n    return 0\n')
            return value
    worker.validation=MutatingValidation()
    result=worker.generate_candidate(workflow(),root=str(tmp_path),diff='real diff')
    assert not result.repair.candidate['selectable']
    assert 'modified production source' in result.repair_attempts[-1]['failure_reason']
    assert 'return a * b' in (tmp_path/'app.py').read_text()

def test_original_regression_failure_does_not_skip_original_integration(tmp_path):
    make_repo(tmp_path)
    (tmp_path/'caller.py').write_text('from app import add\n\ndef checkout():\n    return add(1, 2)\n')
    class Provider(LoopProvider):
        def propose_repair(self, context):
            value=super().propose_repair(context)
            value.regression_test={'path':'tests/ai_generated/test_regression.py','target':'app.add','reason':'Protect addition','source':'from app import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n'}
            value.integration_test={'path':'tests/ai_generated/test_integration.py','target':'caller.checkout','dependency_path':'caller.checkout -> app.add','reason':'Exercise caller','source':'from caller import checkout\n\n\ndef test_checkout():\n    assert checkout() == 3\n'}
            return value
    class LocalValidation(ValidationService):
        def validate(self, root, requested=None, generated_integration_files=None, variant='ORIGINAL', **kwargs):
            return super().validate(root, requested=requested or ['build','unit','regression','targeted_integration'], generated_integration_files=generated_integration_files, variant=variant, **kwargs)
    worker=service(Provider());worker.validation=LocalValidation()
    result=worker.generate_candidate(workflow(),root=str(tmp_path),diff='real diff')
    original=result.candidate_comparison['original']
    assert original['regression_tests']['status']=='FAILED'
    assert original['targeted_integration_tests']['status']=='FAILED'
    assert result.repair.candidate['selectable'], result.repair_attempts
