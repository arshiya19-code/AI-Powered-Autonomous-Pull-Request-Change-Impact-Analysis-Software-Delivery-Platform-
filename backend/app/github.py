"""Authenticated GitHub App routes and signed webhook ingestion."""
from __future__ import annotations
import hashlib, hmac, os, secrets, time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, func, select
from .auth import current_user
from .database import Analysis, AuditEvent, DB, GitHubInstallationIntent, PullRequest, RepositoryConnection
from .services.github import GitHubIntegrationError, GitHubService

router=APIRouter(prefix='/api/github')
service=GitHubService()

def _error(exc: GitHubIntegrationError): raise HTTPException(exc.status_code, {'code':exc.code,'message':str(exc)})
def _best_analysis_for_pr(db, user_id: str, pr) -> 'Analysis | None':
    """Select the most appropriate analysis for a PR.

    Priority order (generic – works for every repository/PR):
    1. Newest COMPLETE analysis whose persisted pr_metadata SHAs match the
       current PR base/head SHAs AND that has a validated selectable candidate.
    2. Newest COMPLETE analysis whose persisted pr_metadata SHAs match the
       current PR base/head SHAs (even if candidate was not selectable).
    3. Newest COMPLETE analysis for this PR regardless of SHA.
    4. Newest analysis of any status (fallback – never None when rows exist).
    """
    from sqlalchemy import select as _sel
    candidates = db.scalars(
        _sel(Analysis)
        .where(Analysis.user_id == user_id, Analysis.pull_request_id == pr.id)
        .order_by(Analysis.created_at.desc())
    ).all()
    if not candidates:
        return None
    base_sha, head_sha = pr.base_sha, pr.head_sha
    def _meta_shas(a):
        result = a.result or {}
        meta = (result.get('analysis') or result).get('pr_metadata') or {}
        return meta.get('base_sha'), meta.get('head_sha')
    def _selectable(a):
        repair = (a.result or {}).get('repair') or {}
        candidate = repair.get('candidate') or {}
        return candidate.get('status') == 'validated' and candidate.get('selectable') is True
    sha_match_selectable = [a for a in candidates if a.status == 'COMPLETE' and _meta_shas(a) == (base_sha, head_sha) and _selectable(a)]
    if sha_match_selectable:
        return sha_match_selectable[0]
    sha_match_complete = [a for a in candidates if a.status == 'COMPLETE' and _meta_shas(a) == (base_sha, head_sha)]
    if sha_match_complete:
        return sha_match_complete[0]
    any_complete = [a for a in candidates if a.status == 'COMPLETE']
    if any_complete:
        return any_complete[0]
    return candidates[0]

def _pr_json(pr, connection, analysis=None):
    return {'id':pr.id,'repository_connection_id':connection.id,'repository':connection.full_name or connection.repository,'pr_number':pr.pr_number,'title':pr.title,'state':pr.state,'author':pr.author,'base_sha':pr.base_sha,'head_sha':pr.head_sha,'base_branch':pr.base_branch,'head_branch':pr.head_branch,'html_url':pr.html_url,'opened_at':pr.opened_at,'updated_at':pr.updated_at,'merged_at':pr.merged_at,'last_synced_at':pr.last_synced_at,'analysis_id':analysis.id if analysis else None,'analysis_status':analysis.status if analysis else 'NOT_ANALYZED','risk':((analysis.result or {}).get('analysis',analysis.result or {}).get('risk_prediction') or {}).get('risk_probability') if analysis else None}

def _upsert_pr(db, connection, payload):
    values=service.pull_request_values(payload)
    row=db.scalar(select(PullRequest).where(PullRequest.repository_connection_id==connection.id,PullRequest.provider_pr_id==values['provider_pr_id']))
    if row:
        for key,value in values.items(): setattr(row,key,value)
    else:
        row=PullRequest(repository_connection_id=connection.id,**values); db.add(row)
    return row

@router.get('/status')
def status(user=Depends(current_user)):
    with DB() as db:
        count=db.scalar(select(func.count()).select_from(RepositoryConnection).where(RepositoryConnection.user_id==user.id,RepositoryConnection.provider=='github',RepositoryConnection.connection_status=='CONNECTED')) or 0
    return {'configured':service.configured,'connected':count>0,'repository_count':count}

@router.post('/connect')
def connect(user=Depends(current_user)):
    state=secrets.token_urlsafe(32)
    with DB() as db:
        db.add(GitHubInstallationIntent(state_hash=hashlib.sha256(state.encode()).hexdigest(),user_id=user.id,expires_at=time.time()+600)); db.commit()
    try: return {'install_url':service.installation_url(state)}
    except GitHubIntegrationError as exc: _error(exc)

@router.get('/callback')
def callback(installation_id: str, state: str, setup_action: str | None=None, user=Depends(current_user)):
    state_hash=hashlib.sha256(state.encode()).hexdigest()
    with DB() as db:
        intent=db.get(GitHubInstallationIntent,state_hash)
        if not intent or intent.user_id!=user.id or intent.expires_at<time.time(): raise HTTPException(400,'GitHub connection state is invalid or expired')
        db.delete(intent); db.commit()
    try: _sync_installation(user.id,str(installation_id))
    except GitHubIntegrationError as exc: _error(exc)
    frontend=os.getenv('FRONTEND_URL','http://localhost:5173').rstrip('/')
    return RedirectResponse(f'{frontend}/#Repositories',status_code=303)

def _sync_installation(user_id: str, installation_id: str):
    repos=service.list_authorized_repositories(installation_id); now=time.time()
    with DB() as db:
        existing={r.repository_id:r for r in db.scalars(select(RepositoryConnection).where(RepositoryConnection.user_id==user_id,RepositoryConnection.provider=='github')).all()}
        seen=set()
        for item in repos:
            values=service.repository_values(item); seen.add(values['repository_id']); row=existing.get(values['repository_id'])
            if row:
                for key,value in values.items(): setattr(row,key,value)
                row.installation_id=installation_id
            else: db.add(RepositoryConnection(user_id=user_id,provider='github',installation_id=installation_id,connected_at=now,analysis_mode='manual',**values))
        for repo_id,row in existing.items():
            if row.installation_id==installation_id and repo_id not in seen: row.connection_status='REVOKED'
        db.add(AuditEvent(actor_id=user_id,action='GITHUB_REPOSITORIES_SYNCED',details={'installation_id':installation_id,'count':len(repos)})); db.commit()
    return len(repos)

@router.get('/repositories')
def repositories(user=Depends(current_user)):
    with DB() as db:
        rows=db.scalars(select(RepositoryConnection).where(RepositoryConnection.user_id==user.id,RepositoryConnection.provider=='github').order_by(RepositoryConnection.connected_at.desc())).all()
        return {'items':[{'id':r.id,'provider':r.provider,'installation_id':r.installation_id,'repository_id':r.repository_id,'owner':r.owner,'repo_name':r.repo_name,'full_name':r.full_name or r.repository,'default_branch':r.default_branch,'connection_status':r.connection_status,'connected_at':r.connected_at,'last_synced_at':r.last_synced_at,'analysis_mode':r.analysis_mode,'open_pr_count':db.scalar(select(func.count()).select_from(PullRequest).where(PullRequest.repository_connection_id==r.id,PullRequest.state=='open')) or 0} for r in rows]}

@router.post('/repositories/{connection_id}/sync')
def sync(connection_id: str,user=Depends(current_user)):
    with DB() as db:
        row=db.scalar(select(RepositoryConnection).where(RepositoryConnection.id==connection_id,RepositoryConnection.user_id==user.id,RepositoryConnection.provider=='github'))
        if not row: raise HTTPException(404,'Repository connection not found')
        installation_id=row.installation_id
    try: count=_sync_installation(user.id,installation_id)
    except GitHubIntegrationError as exc:
        if exc.code=='access_revoked':
            with DB() as db:
                revoked=db.get(RepositoryConnection,connection_id)
                if revoked and revoked.user_id==user.id: revoked.connection_status='REVOKED'; db.commit()
        _error(exc)
    sync_pull_requests(connection_id,user)
    return {'status':'SYNCED','repository_count':count}

@router.patch('/repositories/{connection_id}')
def preference(connection_id: str, body: dict,user=Depends(current_user)):
    mode=body.get('analysis_mode')
    if mode not in ('manual','automatic'): raise HTTPException(422,'analysis_mode must be manual or automatic')
    with DB() as db:
        row=db.scalar(select(RepositoryConnection).where(RepositoryConnection.id==connection_id,RepositoryConnection.user_id==user.id))
        if not row: raise HTTPException(404,'Repository connection not found')
        row.analysis_mode=mode; db.commit()
    return {'analysis_mode':mode}

@router.delete('/repositories/{connection_id}')
def disconnect(connection_id: str,user=Depends(current_user)):
    with DB() as db:
        row=db.scalar(select(RepositoryConnection).where(RepositoryConnection.id==connection_id,RepositoryConnection.user_id==user.id))
        if not row: raise HTTPException(404,'Repository connection not found')
        row.connection_status='DISCONNECTED'; db.add(AuditEvent(actor_id=user.id,action='GITHUB_REPOSITORY_DISCONNECTED',subject_id=row.id)); db.commit()
    return {'status':'DISCONNECTED'}

@router.post('/repositories/{connection_id}/pull-requests/sync')
def sync_pull_requests(connection_id: str,user=Depends(current_user)):
    with DB() as db:
        connection=db.scalar(select(RepositoryConnection).where(RepositoryConnection.id==connection_id,RepositoryConnection.user_id==user.id,RepositoryConnection.connection_status=='CONNECTED'))
        if not connection: raise HTTPException(404,'Connected repository not found')
        installation_id,full_name=connection.installation_id,connection.full_name or connection.repository
    try: items=service.list_open_pull_requests(installation_id,full_name)
    except GitHubIntegrationError as exc:
        if exc.code=='access_revoked':
            with DB() as db:
                revoked=db.get(RepositoryConnection,connection_id)
                if revoked and revoked.user_id==user.id: revoked.connection_status='REVOKED'; db.commit()
        _error(exc)
    with DB() as db:
        connection=db.get(RepositoryConnection,connection_id); seen=set()
        for item in items: seen.add(str(item['id'])); _upsert_pr(db,connection,item)
        for pr in db.scalars(select(PullRequest).where(PullRequest.repository_connection_id==connection_id,PullRequest.state=='open')).all():
            if pr.provider_pr_id not in seen: pr.state='closed'; pr.last_synced_at=time.time()
        connection.last_synced_at=time.time(); db.commit()
    return {'status':'SYNCED','count':len(items)}

@router.get('/pull-requests')
def pull_requests(repository_id: str|None=None,state: str|None=None,risk: str|None=None,analysis_status: str|None=None,date_from: float|None=None,date_to: float|None=None,user=Depends(current_user)):
    with DB() as db:
        connections=db.scalars(select(RepositoryConnection).where(RepositoryConnection.user_id==user.id,RepositoryConnection.provider=='github')).all(); by_id={r.id:r for r in connections}
        query=select(PullRequest).where(PullRequest.repository_connection_id.in_(list(by_id) or ['']))
        if repository_id: query=query.where(PullRequest.repository_connection_id==repository_id)
        if state: query=query.where(PullRequest.state==state)
        if date_from: query=query.where(PullRequest.updated_at>=date_from)
        if date_to: query=query.where(PullRequest.updated_at<=date_to)
        rows=db.scalars(query.order_by(PullRequest.updated_at.desc())).all(); output=[]
        for pr in rows:
            analysis=_best_analysis_for_pr(db,user.id,pr)
            item=_pr_json(pr,by_id[pr.repository_connection_id],analysis)
            if analysis_status and item['analysis_status']!=analysis_status: continue
            if risk=='high' and not (item['risk'] is not None and item['risk']>=.5): continue
            if risk=='low' and not (item['risk'] is not None and item['risk']<.5): continue
            if risk=='unavailable' and item['risk'] is not None: continue
            output.append(item)
        return {'items':output}

@router.get('/pull-requests/{pr_id}/best-analysis')
def best_analysis_for_pr(pr_id: str, user=Depends(current_user)):
    """Return the best analysis for this PR using the SHA-aware selection logic.

    The caller can use the returned analysis_id to load the correct evidence
    without relying on localStorage or a stale URL parameter.
    """
    with DB() as db:
        connections = db.scalars(select(RepositoryConnection).where(RepositoryConnection.user_id == user.id, RepositoryConnection.provider == 'github')).all()
        connection_ids = [c.id for c in connections]
        pr = db.scalar(select(PullRequest).where(PullRequest.id == pr_id, PullRequest.repository_connection_id.in_(connection_ids or [''])))
        if not pr: raise HTTPException(404, 'Pull request not found')
        analysis = _best_analysis_for_pr(db, user.id, pr)
        if not analysis: raise HTTPException(404, 'No analysis found for this pull request')
        repair = (analysis.result or {}).get('repair') or {}
        candidate = repair.get('candidate') or {}
        meta = ((analysis.result or {}).get('analysis') or analysis.result or {}).get('pr_metadata') or {}
        return {
            'analysis_id': analysis.id,
            'status': analysis.status,
            'created_at': analysis.created_at,
            'pr_base_sha': pr.base_sha,
            'pr_head_sha': pr.head_sha,
            'analysis_base_sha': meta.get('base_sha'),
            'analysis_head_sha': meta.get('head_sha'),
            'sha_match': meta.get('base_sha') == pr.base_sha and meta.get('head_sha') == pr.head_sha,
            'candidate_status': candidate.get('status'),
            'candidate_selectable': candidate.get('selectable'),
        }

@router.post('/pull-requests/{pr_id}/analyze',status_code=202)
def analyze_pull_request(pr_id: str,user=Depends(current_user)):
    from .product import create_github_analysis
    return create_github_analysis(pr_id,user)

@router.post('/webhook')
async def webhook(request: Request):
    secret=os.getenv('GITHUB_WEBHOOK_SECRET')
    if not secret: raise HTTPException(503,'GitHub webhook secret is not configured')
    body=await request.body(); supplied=request.headers.get('x-hub-signature-256','')
    expected='sha256='+hmac.new(secret.encode(),body,hashlib.sha256).hexdigest()
    if not hmac.compare_digest(supplied,expected): raise HTTPException(401,'Invalid webhook signature')
    event=request.headers.get('x-github-event'); payload=await request.json()
    if event=='ping': return {'status':'ok'}
    if event!='pull_request': return {'status':'ignored'}
    action=payload.get('action'); supported={'opened','synchronize','reopened','closed'}
    if action not in supported: return {'status':'ignored'}
    installation_id=str((payload.get('installation') or {}).get('id') or '')
    repository_id=str((payload.get('repository') or {}).get('id') or '')
    pr_payload=payload.get('pull_request') or {}; automatic=[]
    with DB() as db:
        connections=db.scalars(select(RepositoryConnection).where(RepositoryConnection.provider=='github',RepositoryConnection.installation_id==installation_id,RepositoryConnection.repository_id==repository_id,RepositoryConnection.connection_status=='CONNECTED')).all()
        for connection in connections:
            pr=_upsert_pr(db,connection,pr_payload); db.flush()
            if connection.analysis_mode=='automatic' and action in {'opened','synchronize','reopened'}: automatic.append((pr.id,connection.user_id))
        db.commit()
    from .product import enqueue_github_analysis
    for pr_id,user_id in automatic: enqueue_github_analysis(pr_id,user_id)
    return {'status':'accepted','action':'merged' if action=='closed' and pr_payload.get('merged') else action,'connections_updated':len(connections)}
