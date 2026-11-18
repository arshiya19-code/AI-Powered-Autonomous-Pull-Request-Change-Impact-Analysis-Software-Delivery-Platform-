"""GitHub App integration, ownership, webhook, and real-PR workflow coverage."""
import hashlib, hmac, json, time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app import auth, database, github, product
from backend.app.main import app

ORIGIN={'origin':'http://localhost:5173'}

@pytest.fixture()
def github_clients(tmp_path,monkeypatch):
    engine=create_engine('sqlite:///'+str(tmp_path/'github.db'),connect_args={'check_same_thread':False})
    database.Base.metadata.create_all(engine); factory=sessionmaker(engine,expire_on_commit=False)
    for module in (auth,github,product): monkeypatch.setattr(module,'DB',factory)
    monkeypatch.setenv('GITHUB_WEBHOOK_SECRET','test-webhook-secret'); auth.attempts.clear()
    first,second=TestClient(app,headers=ORIGIN),TestClient(app,headers=ORIGIN)
    for client,name in ((first,'owner'),(second,'other')):
        assert client.post('/api/auth/register',json={'email':f'{name}@example.test','display_name':name,'password':'a sufficiently long password'}).status_code==201
    yield first,second,factory,tmp_path
    engine.dispose()

def repository_payload(repo_id=101,full_name='team/platform'):
    owner,name=full_name.split('/')
    return {'id':repo_id,'name':name,'full_name':full_name,'default_branch':'main','owner':{'login':owner}}

def pr_payload(pr_id=501,number=7,state='open',merged=False,head='b'*40):
    return {'id':pr_id,'number':number,'title':'Make impact visible','state':state,'merged':merged,'user':{'login':'octocat'},'base':{'sha':'a'*40,'ref':'main'},'head':{'sha':head,'ref':'feature'},'html_url':f'https://github.com/team/platform/pull/{number}','created_at':'2026-09-12T10:00:00Z','updated_at':'2026-09-13T10:00:00Z','merged_at':'2026-09-13T11:00:00Z' if merged else None}

def install(owner,factory,monkeypatch):
    user_id=owner.get('/api/auth/me').json()['id']
    monkeypatch.setattr(github.service,'list_authorized_repositories',lambda installation_id:[repository_payload()])
    assert github._sync_installation(user_id,'9001')==1
    with factory() as db: return db.scalar(select(database.RepositoryConnection).where(database.RepositoryConnection.user_id==user_id))

def test_repository_and_pr_sync_are_owner_scoped(github_clients,monkeypatch):
    owner,other,factory,_=github_clients; connection=install(owner,factory,monkeypatch)
    monkeypatch.setattr(github.service,'list_open_pull_requests',lambda installation_id,full_name:[pr_payload()])
    response=owner.post(f'/api/github/repositories/{connection.id}/pull-requests/sync')
    assert response.status_code==200 and response.json()['count']==1
    assert owner.get('/api/github/repositories').json()['items'][0]['open_pr_count']==1
    assert len(owner.get('/api/github/pull-requests').json()['items'])==1
    assert other.get('/api/github/repositories').json()['items']==[]
    assert other.post(f'/api/github/repositories/{connection.id}/sync').status_code==404
    pr_id=owner.get('/api/github/pull-requests').json()['items'][0]['id']
    assert other.post(f'/api/github/pull-requests/{pr_id}/analyze').status_code==404

def test_connect_uses_expiring_owner_bound_state(github_clients,monkeypatch):
    owner,other,factory,_=github_clients
    monkeypatch.setattr(github.service,'installation_url',lambda state:f'https://github.com/apps/test/installations/new?state={state}')
    result=owner.post('/api/github/connect').json(); state=result['install_url'].split('state=')[1]
    monkeypatch.setattr(github.service,'list_authorized_repositories',lambda installation_id:[repository_payload()])
    assert other.get('/api/github/callback',params={'installation_id':'9001','state':state}).status_code==400
    assert owner.get('/api/github/callback',params={'installation_id':'9001','state':state},follow_redirects=False).status_code==303
    assert owner.get('/api/github/callback',params={'installation_id':'9001','state':state},follow_redirects=False).status_code==400

def signed(payload: dict,secret='test-webhook-secret'):
    body=json.dumps(payload,separators=(',',':')).encode()
    return body,{'content-type':'application/json','x-github-event':'pull_request','x-hub-signature-256':'sha256='+hmac.new(secret.encode(),body,hashlib.sha256).hexdigest()}

def test_webhook_signature_and_pr_lifecycle(github_clients,monkeypatch):
    owner,_,factory,_=github_clients; connection=install(owner,factory,monkeypatch)
    payload={'action':'opened','installation':{'id':9001},'repository':repository_payload(),'pull_request':pr_payload()}
    body,headers=signed(payload)
    anonymous=TestClient(app)
    assert anonymous.post('/api/github/webhook',content=body,headers={**headers,'x-hub-signature-256':'sha256=bad'}).status_code==401
    assert anonymous.post('/api/github/webhook',content=body,headers=headers).json()['action']=='opened'
    payload['action']='synchronize'; payload['pull_request']['head']['sha']='c'*40
    body,headers=signed(payload); assert anonymous.post('/api/github/webhook',content=body,headers=headers).status_code==200
    payload['action']='closed'; payload['pull_request'].update({'state':'closed','merged':True,'merged_at':'2026-09-13T11:00:00Z'})
    body,headers=signed(payload); assert anonymous.post('/api/github/webhook',content=body,headers=headers).json()['action']=='merged'
    with factory() as db:
        pr=db.scalar(select(database.PullRequest).where(database.PullRequest.repository_connection_id==connection.id))
        assert pr.state=='merged' and pr.head_sha=='c'*40 and pr.merged_at

def test_manual_and_automatic_analysis_use_isolated_owned_workspace(github_clients,monkeypatch):
    owner,other,factory,tmp_path=github_clients; connection=install(owner,factory,monkeypatch)
    with factory() as db:
        current=db.get(database.RepositoryConnection,connection.id); pr=database.PullRequest(repository_connection_id=current.id,**github.service.pull_request_values(pr_payload())); db.add(pr); db.commit(); pr_id=pr.id
    observed=[]
    diff='diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n@@ -1,2 +1,2 @@\n def changed():\n-    return 1\n+    return 2\n'
    def materialize(installation_id,full_name,base_sha,head_sha,workspace):
        assert base_sha=='d'*40 and head_sha=='e'*40
        root=Path(workspace); observed.append(root); (root/'sample.py').write_text('def changed():\n    return 2\n')
        return diff,{'sample.py':'def changed():\n    return 2\n'},[]
    monkeypatch.setattr(github.service,'materialize_pull_request',materialize)
    live=pr_payload()
    live.update(title='Calculator.percent expected behavior',body='Expected: 50% = 0.5; Actual defect: 50% = 5000')
    live['base']={'sha':'d'*40,'ref':'demo-risk-change'}
    live['head']={'sha':'e'*40,'ref':'demo-live-ai-repair'}
    monkeypatch.setattr(github.service,'get_pull_request',lambda *args:live)
    class Immediate:
        def submit(self,fn,*args): fn(*args)
    monkeypatch.setattr(product,'executor',Immediate())
    created=owner.post(f'/api/github/pull-requests/{pr_id}/analyze')
    assert created.status_code==202; analysis_id=created.json()['id']
    detail=owner.get(f'/api/product/analyses/{analysis_id}').json()
    assert detail['status']=='COMPLETE' and detail['result']['graph']['nodes']
    assert detail['repository_connection_id']==connection.id and detail['pull_request_id']==pr_id
    assert all(not path.exists() for path in observed)
    assert other.get(f'/api/product/analyses/{analysis_id}').status_code==404
    assert owner.post('/api/product/analyses',json={'repository_path':str(tmp_path),'diff':diff}).status_code==422
    with factory() as db:
        row=db.get(database.Analysis,analysis_id)
        assert row.user_id==owner.get('/api/auth/me').json()['id']
        metadata=row.result['analysis']['pr_metadata']
        assert metadata['base_sha']=='d'*40 and metadata['head_sha']=='e'*40
        assert metadata['title']==live['title'] and metadata['body']==live['body']
        assert (row.result or {}).get('approval_state')!='DEPLOYMENT_COMPLETED'

def test_automatic_mode_enqueues_only_open_sync_events(github_clients,monkeypatch):
    owner,_,factory,_=github_clients; connection=install(owner,factory,monkeypatch)
    with factory() as db: db.get(database.RepositoryConnection,connection.id).analysis_mode='automatic'; db.commit()
    calls=[]; monkeypatch.setattr(product,'enqueue_github_analysis',lambda pr_id,user_id:calls.append((pr_id,user_id)))
    payload={'action':'opened','installation':{'id':9001},'repository':repository_payload(),'pull_request':pr_payload()}
    body,headers=signed(payload); assert TestClient(app).post('/api/github/webhook',content=body,headers=headers).status_code==200
    assert len(calls)==1
    payload['action']='closed'; body,headers=signed(payload); TestClient(app).post('/api/github/webhook',content=body,headers=headers)
    assert len(calls)==1

def test_admin_can_see_github_analysis_but_normal_user_cannot(github_clients,monkeypatch):
    owner,other,factory,_=github_clients; connection=install(owner,factory,monkeypatch); owner_id=owner.get('/api/auth/me').json()['id']; other_id=other.get('/api/auth/me').json()['id']
    with factory() as db:
        pr=database.PullRequest(repository_connection_id=connection.id,**github.service.pull_request_values(pr_payload())); db.add(pr); db.flush()
        analysis=database.Analysis(user_id=owner_id,repository='team/platform',pull_request=7,repository_connection_id=connection.id,pull_request_id=pr.id,status='COMPLETE',result={}); db.add(analysis)
        db.get(database.User,other_id).role='ADMIN'; db.commit(); analysis_id=analysis.id
    assert other.get('/api/admin/analyses').status_code==200
    assert any(x['id']==analysis_id for x in other.get('/api/admin/analyses').json()['items'])
    assert other.get(f'/api/product/analyses/{analysis_id}').status_code==404
