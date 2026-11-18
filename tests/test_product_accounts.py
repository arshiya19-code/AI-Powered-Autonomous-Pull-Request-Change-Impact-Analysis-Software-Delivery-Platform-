"""Security boundary and real-analysis integration coverage."""
import time
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from backend.app import database, auth, product
from backend.app.main import app

ORIGIN = {'origin': 'http://localhost:5173'}

@pytest.fixture()
def clients(tmp_path, monkeypatch):
    engine = create_engine('sqlite:///' + str(tmp_path / 'accounts.db'), connect_args={'check_same_thread': False})
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(auth, 'DB', factory); monkeypatch.setattr(product, 'DB', factory)
    auth.attempts.clear()
    a, b = TestClient(app, headers=ORIGIN), TestClient(app, headers=ORIGIN)
    for c, name in [(a,'first'),(b,'second')]:
        r=c.post('/api/auth/register',json={'email':f'{name}@example.test','display_name':name,'password':'a long private password'})
        assert r.status_code == 201, r.text
        assert 'password' not in r.text
        assert 'HttpOnly' in r.headers['set-cookie']
        assert 'SameSite=lax' in r.headers['set-cookie']
    yield a,b,factory
    engine.dispose()

def test_sessions_password_and_csrf(clients):
    a,b,db=clients
    assert a.get('/api/auth/me').status_code==200
    assert a.post('/api/auth/refresh').status_code==200
    assert a.patch('/api/auth/profile',headers={'origin':'https://evil.test'},json={'display_name':'wrong'}).status_code==403
    assert a.patch('/api/auth/profile',json={'display_name':'Updated','preferences':{'theme':'night'}}).status_code==200
    assert a.post('/api/auth/password',json={'current_password':'wrong','password':'another long password'}).status_code==400
    assert a.post('/api/auth/password',json={'current_password':'a long private password','password':'another long password'}).status_code==200
    with db() as s:
        user=s.scalar(select(database.User).where(database.User.email=='first@example.test'))
        assert user.password_hash.startswith('$argon2id$')
    assert a.post('/api/auth/logout').status_code==200
    assert a.get('/api/auth/me').status_code==401
    assert a.post('/api/auth/login',json={'email':'first@example.test','password':'a long private password'}).status_code==401
    assert a.post('/api/auth/login',json={'email':'first@example.test','password':'another long password'}).status_code==200

def test_isolation_admin_and_legacy(clients):
    a,b,db=clients
    uid=a.get('/api/auth/me').json()['id']; other=b.get('/api/auth/me').json()['id']
    with db() as s:
        row=database.Analysis(user_id=uid,repository='private',status='COMPLETE',result={'private':'evidence'},events=[])
        s.add(row);s.commit(); aid=row.id
    assert b.get('/api/product/analyses').json()['items']==[]
    assert b.get(f'/api/product/analyses/{aid}').status_code==404
    assert b.get(f'/api/product/analyses/{aid}/events').status_code==404
    assert b.get('/api/admin/users').status_code==403
    assert b.get('/api/v1/history').status_code==403
    assert b.post('/api/deployment/someone-else/approve').status_code==403
    assert a.get(f'/api/product/analyses/{aid}').json()['result']['private']=='evidence'
    with db() as s:
        s.get(database.User,uid).role='ADMIN';s.commit()
    users=a.get('/api/admin/users');assert users.status_code==200
    assert 'password_hash' not in users.text and 'token_hash' not in users.text
    assert a.patch(f'/api/admin/users/{uid}',json={'role':'USER','is_active':False}).status_code==409
    assert a.patch(f'/api/admin/users/{other}',json={'role':'USER','is_active':False}).status_code==200
    assert b.get('/api/auth/me').status_code==401

def test_real_analysis_and_events(clients):
    a,b,db=clients
    source='def changed():\n    return 2\n\ndef caller():\n    return changed()\n'
    diff='diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n@@ -1,2 +1,2 @@\n def changed():\n-    return 1\n+    return 2\n'
    assert a.post('/api/product/analyses',json={'repository_path':'C:/private','diff':diff}).status_code==422
    r=a.post('/api/product/analyses',json={'repository':'test/repo','diff':diff,'sources':{'sample.py':source}})
    assert r.status_code==202,r.text
    aid=r.json()['id']
    for _ in range(600):
        result=a.get(f'/api/product/analyses/{aid}').json()
        if result['status'] in ('COMPLETE','FAILED'):break
        time.sleep(.1)
    assert result['status']=='COMPLETE',result
    assert result['result']['graph']['nodes']
    assert any(n['changed'] for n in result['result']['graph']['nodes'])
    kinds=[e['type'] for e in result['events']]
    assert 'GRAPH_BUILT' in kinds and 'IMPACT_ANALYZED' in kinds
    assert b.get(f'/api/product/analyses/{aid}').status_code==404
    stream=a.get(f'/api/product/analyses/{aid}/events')
    assert stream.status_code==200 and 'ANALYSIS_COMPLETE' in stream.text
    assert a.get(f'/api/product/analyses/{aid}/events',headers={'Last-Event-ID':'999'}).text==''

def test_validation_and_rate_limit(clients):
    a,b,db=clients
    assert a.post('/api/auth/register',json={'email':'bad','display_name':'X','password':'short'}).status_code==422
    auth.attempts.clear()
    for _ in range(15):
        assert a.post('/api/auth/login',json={'email':'missing@example.test','password':'wrong'}).status_code==401
    assert a.post('/api/auth/login',json={'email':'missing@example.test','password':'wrong'}).status_code==429
def test_reset_tokens_expiry_and_session_revocation(clients):
    a,b,factory=clients
    uid=a.get('/api/auth/me').json()['id']
    with factory() as db:
        db.add(database.PasswordReset(token_hash=auth.digest('test-reset-token'),user_id=uid,expires_at=time.time()+60));db.commit()
    assert b.post('/api/auth/reset-password',json={'token':'bad','password':'a new private passphrase'}).status_code==400
    assert b.post('/api/auth/reset-password',json={'token':'test-reset-token','password':'a new private passphrase'}).status_code==200
    assert a.get('/api/auth/me').status_code==401
    assert b.post('/api/auth/reset-password',json={'token':'test-reset-token','password':'a new private passphrase'}).status_code==400
    assert a.post('/api/auth/login',json={'email':'first@example.test','password':'a new private passphrase'}).status_code==200

def test_approval_is_owner_scoped_and_requires_validation(clients):
    from backend.app.schemas.models import WorkflowResponse, DeploymentSimulationResult, DecisionResult
    a,b,factory=clients
    uid=a.get('/api/auth/me').json()['id']
    workflow=WorkflowResponse(workflow_id='test-workflow',approval_state='READY_FOR_APPROVAL',deployment=DeploymentSimulationResult(recommended_action='DEPLOY'),decision=DecisionResult(decision='DEPLOY'))
    with factory() as db:
        row=database.Analysis(user_id=uid,repository='private',status='COMPLETE',result=workflow.model_dump(mode='json'));db.add(row);db.commit();aid=row.id
    assert b.post(f'/api/product/analyses/{aid}/approve').status_code==404
    assert a.post(f'/api/product/analyses/{aid}/approve').status_code==409
    with factory() as db:
        assert db.get(database.Analysis,aid).result['approval_state']=='READY_FOR_APPROVAL'

def test_source_traversal_and_anonymous_access(clients):
    a,b,factory=clients
    anonymous=TestClient(app)
    assert anonymous.get('/api/product/analyses').status_code==401
    for path in ['../private.py','C:/private.py','folder\\..\\private.py','/absolute.py']:
        assert a.post('/api/product/analyses',json={'diff':'a diff','sources':{path:'x'}}).status_code==422

def test_admin_sees_persisted_user_activity_end_to_end(clients):
    admin_client,user_client,factory=clients
    admin_id=admin_client.get('/api/auth/me').json()['id']
    normal_user=user_client.get('/api/auth/me').json()
    with factory() as db:
        db.get(database.User,admin_id).role='ADMIN';db.commit()
    source='def changed():\n    return 2\n'
    diff='diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n@@ -1,2 +1,2 @@\n def changed():\n-    return 1\n+    return 2\n'
    created=user_client.post('/api/product/analyses',json={'repository':'team/live-platform','pull_request':42,'diff':diff,'sources':{'sample.py':source}})
    assert created.status_code==202,created.text
    analysis_id=created.json()['id']
    for _ in range(600):
        persisted=user_client.get(f'/api/product/analyses/{analysis_id}').json()
        if persisted['status'] in ('COMPLETE','FAILED'):break
        time.sleep(.1)
    assert persisted['status']=='COMPLETE'
    with factory() as db:
        assert db.get(database.Analysis,analysis_id).user_id==normal_user['id']
    overview=admin_client.get('/api/admin/overview')
    assert overview.status_code==200
    assert overview.json()['stats']['total_analyses']>=1
    assert overview.json()['stats']['completed_analyses']>=1
    users=admin_client.get('/api/admin/users').json()['items']
    visible_user=next(item for item in users if item['id']==normal_user['id'])
    assert visible_user['analysis_count']==1
    detail=admin_client.get(f"/api/admin/users/{normal_user['id']}").json()
    assert detail['usage']['recent_analyses'][0]['id']==analysis_id
    analyses=admin_client.get('/api/admin/analyses',params={'user_id':normal_user['id'],'repository':'live-platform'}).json()['items']
    assert analyses[0]['id']==analysis_id and analyses[0]['pull_request']==42
    activity=admin_client.get('/api/admin/activity').json()['items']
    assert any(item['action']=='ANALYSIS_CREATED' and item['subject_id']==analysis_id for item in activity)
    assert any(item['action']=='ANALYSIS_COMPLETE' and item['subject_id']==analysis_id for item in activity)
    assert user_client.get('/api/admin/overview').status_code==403
    assert admin_client.get(f'/api/product/analyses/{analysis_id}').status_code==404
