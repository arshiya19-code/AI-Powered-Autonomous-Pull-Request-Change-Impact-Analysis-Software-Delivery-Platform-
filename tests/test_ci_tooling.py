import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from backend.app.schemas.models import ValidationCheck
from backend.app.services.ci_tools import ScannerService, redact, command_env
from backend.app.services.jenkins import JenkinsService, STAGES
from backend.app.services.validation import ValidationService
from backend.app.services.github import GitHubService


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in ('JENKINS_URL','JENKINS_JOB','JENKINS_USER','JENKINS_TOKEN','CI_SHARED_ROOT','SONAR_HOST_URL','SONAR_TOKEN','SNYK_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CI_EXECUTOR','local')
    monkeypatch.setattr('backend.app.services.ci_tools.executable', lambda name: None)


def configured_jenkins(monkeypatch, tmp_path, handler):
    for key,value in {'JENKINS_URL':'http://jenkins.test','JENKINS_JOB':'candidate-validation','JENKINS_USER':'ci','JENKINS_TOKEN':'secret-jenkins-test','CI_SHARED_ROOT':str(tmp_path/'bundles')}.items():
        monkeypatch.setenv(key,value)
    return JenkinsService(httpx.Client(transport=httpx.MockTransport(handler)))


def identity():
    return {'analysis_id':'analysis-2','candidate_id':'candidate-2','attempt':2,'bundle_sha256':'digest','base_sha':'a'*40,'head_sha':'b'*40}


def results():
    checks={k:ValidationCheck(check_type=k,status='PASSED',variant='AI_CANDIDATE').model_dump() for k in STAGES}
    checks['sonarqube']['evidence']={'quality_gate':'OK'}
    checks['snyk']['evidence']={'baseline_compared':True,'introduced':{'critical':0,'high':0,'medium':2,'low':0}}
    return {**identity(),'checks':checks,'handoff':'BACKEND_REANALYSIS_REQUIRED'}


def test_jenkins_unconfigured():
    assert JenkinsService().readiness()['status']=='NOT CONFIGURED'


def test_jenkins_unreachable(monkeypatch,tmp_path):
    service=configured_jenkins(monkeypatch,tmp_path,lambda request:httpx.Response(503))
    assert service.readiness()['status']=='UNREACHABLE'


def test_jenkins_ready_and_no_secrets(monkeypatch,tmp_path):
    service=configured_jenkins(monkeypatch,tmp_path,lambda request:httpx.Response(200,json={'buildable':True}))
    assert service.readiness()['status']=='READY'
    assert 'secret-jenkins-test' not in json.dumps(service.readiness())


@pytest.mark.parametrize('field,value',[('analysis_id','other'),('candidate_id','other'),('attempt',3),('bundle_sha256','other'),('base_sha','other'),('head_sha','other')])
def test_jenkins_rejects_wrong_correlation(field,value):
    payload=results(); payload[field]=value
    with pytest.raises(ValueError,match='identity mismatch'): JenkinsService.ingest(payload,identity())


def test_jenkins_ingestion_recomputes_gate():
    payload=results(); payload['checks']['snyk']['evidence']['introduced']['high']=1
    summary=JenkinsService.ingest(payload,identity())
    assert summary.status=='failed' and 'snyk' in summary.blocking_failures
    assert summary.ci['analysis_id']=='analysis-2'


def test_jenkins_missing_stage_rejected():
    payload=results(); del payload['checks']['unit']
    with pytest.raises(ValueError,match='incomplete'): JenkinsService.ingest(payload,identity())


def test_jenkins_missing_tools_are_not_passes():
    payload=results()
    for key in ('sonarqube','snyk'):
        payload['checks'][key]=ValidationCheck(check_type=key,status='UNAVAILABLE',variant='AI_CANDIDATE',summary='NOT CONFIGURED').model_dump()
    summary=JenkinsService.ingest(payload,identity())
    assert summary.checks['sonarqube']['status']=='UNAVAILABLE'
    assert summary.checks['snyk']['status']=='UNAVAILABLE'


@pytest.mark.parametrize('terminal',['SUCCESS','FAILURE','ABORTED'])
def test_jenkins_submission_polling_and_artifact(monkeypatch,tmp_path,terminal):
    root=tmp_path/'source'; root.mkdir(); (root/'app.py').write_text('value=1')
    (root/'.env').write_text('SNYK_TOKEN=never-ship-this')
    def handler(request):
        if request.method=='POST':
            return httpx.Response(201,headers={'Location':'http://jenkins.test/queue/item/1/'})
        if '/queue/' in request.url.path:
            return httpx.Response(200,json={'executable':{'url':'http://jenkins.test/job/candidate-validation/1/'}})
        if '/artifact/' in request.url.path:
            manifest=json.loads(next((tmp_path/'bundles').glob('*/manifest.json')).read_text())
            return httpx.Response(200,json={**results(),**manifest,'checks':results()['checks'],'handoff':'BACKEND_REANALYSIS_REQUIRED'})
        return httpx.Response(200,json={'buildable':True,'building':False,'result':terminal})
    service=configured_jenkins(monkeypatch,tmp_path,handler)
    monkeypatch.setattr('backend.app.services.jenkins.time.sleep',lambda _:None)
    summary=service.validate(root,root,identity(),['tests/test_regression.py'],[])
    assert summary.ci['status']==('PASS' if terminal=='SUCCESS' else 'FAIL')
    assert not list((tmp_path/'bundles').rglob('.env'))
    assert summary.ci['analysis_id']=='analysis-2'


def test_jenkins_timeout_blocks_candidate(monkeypatch,tmp_path):
    root=tmp_path/'source'; root.mkdir()
    service=configured_jenkins(monkeypatch,tmp_path,lambda request:httpx.Response(201,headers={'Location':'http://jenkins.test/queue/item/1/'},json={'buildable':True}))
    monkeypatch.setenv('JENKINS_TIMEOUT','0')
    result=service.validate(root,root,identity(),[],[])
    assert result.ci['status']=='RUNNING' and result.status=='failed'


def test_jenkins_no_implicit_local_fallback(monkeypatch,tmp_path):
    monkeypatch.setenv('CI_EXECUTOR','jenkins')
    result=ValidationService().validate(tmp_path,variant='AI_CANDIDATE',ci_context=identity(),baseline_root=tmp_path)
    assert result.status=='failed' and result.ci['status']=='NOT CONFIGURED'


def sonar_config(monkeypatch):
    monkeypatch.setenv('SONAR_HOST_URL','http://sonar.test')
    monkeypatch.setenv('SONAR_TOKEN','secret-sonar-test')
    monkeypatch.setattr('backend.app.services.ci_tools.executable',lambda name:name)


def test_sonar_unconfigured(tmp_path):
    service=ScannerService()
    assert service.sonar_readiness()['status']=='NOT CONFIGURED'
    assert service.sonar(tmp_path).status=='UNAVAILABLE'


@pytest.mark.parametrize('gate,status',[('OK','PASSED'),('ERROR','FAILED'),('NONE','ERROR')])
def test_sonar_quality_gate_and_actual_metrics(monkeypatch,tmp_path,gate,status):
    sonar_config(monkeypatch)
    def handler(request):
        if 'authentication' in request.url.path: return httpx.Response(200,json={'valid':True})
        if '/ce/' in request.url.path: return httpx.Response(200,json={'task':{'status':'SUCCESS','analysisId':'scan-2'}})
        if 'qualitygates' in request.url.path:
            assert request.url.params['analysisId']=='scan-2'
            return httpx.Response(200,json={'projectStatus':{'status':gate}})
        return httpx.Response(200,json={'component':{'measures':[{'metric':'bugs','value':'1'},{'metric':'security_hotspots','value':'2'}]}})
    def run(command,**kwargs):
        assert 'secret-sonar-test' not in str(command)
        (tmp_path/'.scannerwork').mkdir(); (tmp_path/'.scannerwork/report-task.txt').write_text('ceTaskId=task-2')
        return SimpleNamespace(returncode=0 if gate=='OK' else 1,stdout='',stderr='')
    check=ScannerService(httpx.Client(transport=httpx.MockTransport(handler)),run).sonar(tmp_path)
    assert check.status==status and check.evidence['measures']['bugs']==1
    assert 'coverage' not in check.evidence['measures']
    assert 'secret-sonar-test' not in check.model_dump_json()
    if status=='FAILED': assert 'sonarqube' in ValidationService().aggregate({'sonarqube':check}).blocking_failures


def test_sonar_unreachable(monkeypatch,tmp_path):
    sonar_config(monkeypatch)
    check=ScannerService(httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(503)))).sonar(tmp_path)
    assert check.status=='ERROR'


def snyk_config(monkeypatch):
    monkeypatch.setenv('SNYK_TOKEN','secret-snyk-test')
    monkeypatch.setattr('backend.app.services.ci_tools.executable',lambda name:name)


def test_snyk_unconfigured(tmp_path):
    assert ScannerService().snyk_readiness()['status']=='NOT CONFIGURED'
    assert ScannerService().snyk(tmp_path).status=='UNAVAILABLE'


def test_snyk_does_not_claim_pass_from_unbound_host_evidence(monkeypatch, tmp_path):
    candidate = tmp_path / 'candidate'
    baseline = tmp_path / 'baseline'
    candidate.mkdir()
    baseline.mkdir()
    (candidate / 'requirements.txt').write_text('flask==1.1.2\n')
    (baseline / 'requirements.txt').write_text('flask==1.1.2\n')
    evidence = tmp_path / 'snyk-host.json'
    evidence.write_text(json.dumps({'vulnerabilities': []}))
    monkeypatch.setenv('SNYK_HOST_EVIDENCE_PATH', str(evidence))
    monkeypatch.setattr('backend.app.services.ci_tools.executable', lambda name: None)

    check = ScannerService().snyk(candidate, baseline)

    assert check.status == 'UNAVAILABLE'
    assert check.evidence['state'] == 'NOT CONFIGURED'


@pytest.mark.parametrize('severity,blocked',[('critical',True),('high',True),('medium',False),('low',False)])
def test_snyk_introduced_gate(monkeypatch,tmp_path,severity,blocked):
    snyk_config(monkeypatch)
    def run(command,**kwargs):
        assert 'secret-snyk-test' not in str(command)
        vulns=[{'id':'SNYK-1','severity':severity,'packageName':'example','version':'1','from':['project','example@1']}] if kwargs['cwd']==tmp_path else []
        return SimpleNamespace(returncode=1 if vulns else 0,stdout=json.dumps({'vulnerabilities':vulns}),stderr='')
    check=ScannerService(run=run).snyk(tmp_path,tmp_path/'baseline')
    assert (check.status=='FAILED')==blocked
    assert check.severity_counts[severity]==1 and check.evidence['introduced'][severity]==1
    assert 'secret-snyk-test' not in check.model_dump_json()


def test_snyk_preexisting_high_is_not_introduced(monkeypatch,tmp_path):
    snyk_config(monkeypatch)
    service=ScannerService(run=lambda *args,**kwargs:SimpleNamespace(returncode=1,stdout=json.dumps({'vulnerabilities':[{'id':'SNYK-1','severity':'high'}]})))
    check=service.snyk(tmp_path,tmp_path/'baseline')
    assert check.status=='PASSED' and check.evidence['introduced']['high']==0


@pytest.mark.parametrize('output,code',[('{}',0),('not json',1),('[]',0),('{"vulnerabilities":[]}',2)])
def test_snyk_errors_cannot_pass(monkeypatch,tmp_path,output,code):
    snyk_config(monkeypatch)
    check=ScannerService(run=lambda *args,**kwargs:SimpleNamespace(returncode=code,stdout=output)).snyk(tmp_path,tmp_path/'baseline')
    assert check.status=='ERROR'


def test_secrets_filtered_from_process_and_output(monkeypatch):
    for key in ('SONAR_TOKEN','SNYK_TOKEN','JENKINS_TOKEN','GITHUB_WEBHOOK_SECRET','LLM_API_KEY'):
        monkeypatch.setenv(key,'sensitive-'+key)
    env=command_env()
    assert not any(key in env for key in ('SONAR_TOKEN','SNYK_TOKEN','JENKINS_TOKEN','GITHUB_WEBHOOK_SECRET','LLM_API_KEY'))
    assert 'sensitive-' not in redact('sensitive-SONAR_TOKEN sensitive-SNYK_TOKEN sensitive-LLM_API_KEY')


def test_stacked_pr_materialization_uses_github_shas(monkeypatch,tmp_path):
    base='a'*40; head='b'*40
    payload={'id':2,'number':2,'title':'Percent defect','base':{'sha':base,'ref':'demo-risk-change'},'head':{'sha':head,'ref':'demo-live-ai-repair'}}
    values=GitHubService.pull_request_values(payload)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if 'checkout' in command: (tmp_path/'calculator.py').write_text('def percent(): return 50*100')
        return SimpleNamespace(returncode=0,stdout='diff --git a/calculator.py b/calculator.py\n+def percent(): return 50*100' if 'diff' in command else '',stderr='')
    monkeypatch.setattr('backend.app.services.github.subprocess.run',run)
    service=GitHubService(); monkeypatch.setattr(service,'installation_token',lambda _: 'fake-installation-token')
    service.materialize_pull_request('1','owner/repo',values['base_sha'],values['head_sha'],tmp_path)
    assert ['git','diff','--no-ext-diff','--binary',base,head] in calls
    assert ['git','checkout','--quiet','--detach',head] in calls
    assert not any(branch in call for call in calls for branch in ('main','master','demo-risk-change'))


def test_qwen_http_payload_contains_real_pr_defect(monkeypatch):
    from tests.test_ai_repair_validation_loop import workflow
    from backend.app.services.workflow import DeliveryWorkflowService
    from backend.app.services.repair import ConfiguredProvider
    result=workflow(); result.analysis.pr_metadata.title='Repair Calculator.percent'
    result.analysis.pr_metadata.body='Expected: 50% = 0.5\nActual defect: 50% = 5000'
    result.analysis.change_details['changed_symbols']=[{'name':'Calculator.percent'}]
    context=DeliveryWorkflowService.repair_context(result,diff='- self._append("/100")\n+ self._append("*100")')
    seen={}
    class Response:
        headers={'Content-Type':'application/json'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def getcode(self): return 200
        def read(self): return json.dumps({'choices':[{'message':{'content':json.dumps({'explanation':'test fixture','confidence':0.5,'affected_files':[],'patch':''})}}]}).encode()
    def urlopen(request,**kwargs):
        seen.update(json.loads(request.data)); return Response()
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    ConfiguredProvider('qwen','qwen-test','http://provider.test/v1').propose_repair(context)
    prompt='\n'.join(message['content'] for message in seen['messages'])
    for term in ('Repair Calculator.percent','50% = 0.5','50% = 5000','*100','/100','changed_symbols','impact','Passing existing tests does not establish correctness'):
        assert term in prompt

def test_python_suites_are_executed_independently(tmp_path):
    (tmp_path/'test_existing.py').write_text('def test_existing():\n    assert 1 + 1 == 2\n')
    (tmp_path/'test_regression.py').write_text('def test_percent():\n    assert 50 * 100 == 0.5\n')
    (tmp_path/'test_integration.py').write_text('def test_caller():\n    assert str(50 / 100) == "0.5"\n')
    service=ValidationService()
    kwargs={'generated_test_files':['test_regression.py'],'generated_integration_files':['test_integration.py'],'variant':'AI_CANDIDATE','ci_context':identity()}
    unit=service.validate(tmp_path,requested=['unit'],**kwargs)
    regression=service.validate(tmp_path,requested=['regression'],**kwargs)
    integration=service.validate(tmp_path,requested=['targeted_integration'],**kwargs)
    assert unit.status=='passed' and regression.status=='failed' and integration.status=='passed'
    assert all(r.ci['analysis_id']=='analysis-2' for r in (unit,regression,integration))
    assert '1 passed' in unit.checks['unit']['stdout_excerpt']


def test_sonar_configured_but_missing_scanner_blocks(monkeypatch,tmp_path):
    monkeypatch.setenv('SONAR_HOST_URL','http://sonar.test')
    monkeypatch.setenv('SONAR_TOKEN','secret-sonar-test')
    assert ScannerService().sonar(tmp_path).status=='ERROR'


def test_dotenv_precedes_backend_configuration_and_does_not_override_process(tmp_path):
    # Isolated package copy avoids writing any real project credentials.
    package=tmp_path/'backend'/'app'; package.mkdir(parents=True)
    (tmp_path/'backend'/'__init__.py').write_text('')
    source=Path(__file__).parents[1]/'backend/app/__init__.py'
    (package/'__init__.py').write_text(source.read_text(encoding='utf-8'))
    (tmp_path/'.env').write_text('CI_STARTUP_TEST=from-file\nCI_EXISTING_TEST=file\n')
    import os,sys
    env=command_env(); env['CI_EXISTING_TEST']='process'; env['PYTHONPATH']=str(tmp_path)
    result=subprocess.run([sys.executable,'-c','import backend.app,os; assert os.environ["CI_STARTUP_TEST"]=="from-file"; assert os.environ["CI_EXISTING_TEST"]=="process"'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_snyk_absolute_manifest_paths_and_root_version_match(monkeypatch,tmp_path):
    snyk_config(monkeypatch)
    def run(*args,**kwargs):
        root=kwargs['cwd']
        report={'targetFile':str(root/'requirements.txt'),'vulnerabilities':[{'id':'SNYK-1','severity':'high','from':['root@'+('2' if root==tmp_path else '1'),'package@1']}]}
        return SimpleNamespace(returncode=1,stdout=json.dumps(report))
    check=ScannerService(run=run).snyk(tmp_path,tmp_path/'baseline')
    assert check.status=='PASSED' and check.evidence['introduced']['high']==0


def test_jenkins_worker_prepares_and_hands_off_correlated_evidence(monkeypatch,tmp_path):
    from scripts.jenkins_candidate import main
    from backend.app.services.jenkins import bundle_digest
    import sys
    bundle_id='12345678-1234-1234-1234-123456789abc'
    bundle=tmp_path/'shared'/bundle_id
    for folder in ('candidate','baseline'):
        (bundle/folder).mkdir(parents=True); (bundle/folder/'app.py').write_text('value=1')
    digest=bundle_digest(bundle)
    manifest={**identity(),'bundle_sha256':digest,'regression':['test_regression.py'],'targeted_integration':[]}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    workspace=tmp_path/'agent'; workspace.mkdir()
    for k,v in {'CI_SHARED_ROOT':str(tmp_path/'shared'),'WORKSPACE':str(workspace),'BUNDLE_ID':bundle_id,'BUNDLE_SHA256':digest}.items(): monkeypatch.setenv(k,v)
    monkeypatch.setattr(sys,'argv',['worker','prepare']); main()
    assert (workspace/'candidate/app.py').read_text()=='value=1'
    monkeypatch.setattr(sys,'argv',['worker','handoff'])
    with pytest.raises(ValueError,match='Missing'): main()
    payload=json.loads((workspace/'ci-result.json').read_text()); payload['checks']=results()['checks']
    (workspace/'ci-result.json').write_text(json.dumps(payload))
    main()
    payload=json.loads((workspace/'ci-result.json').read_text())
    summary=JenkinsService.ingest(payload,manifest)
    assert summary.ci['analysis_id']=='analysis-2' and summary.status=='passed'


def test_secret_file_ignore_patterns():
    import pathspec
    patterns=(Path(__file__).parents[1]/'.gitignore').read_text(encoding='utf-8-sig')
    spec=pathspec.GitIgnoreSpec.from_lines(patterns.splitlines())
    for name in ('.env','.env.local','app.pem','key.key','secrets/token.txt','token.secret'):
        assert spec.match_file(name),name
    assert not spec.match_file('.env.example')
