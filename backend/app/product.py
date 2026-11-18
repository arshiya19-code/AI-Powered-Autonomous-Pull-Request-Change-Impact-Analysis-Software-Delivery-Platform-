"""User-owned analysis jobs and durable SSE events around the existing engine."""
import asyncio
import json
import time
import re
import subprocess
import logging
import traceback
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from .auth import current_user, digest, COOKIE
from .database import DB, Analysis, AuditEvent, PullRequest, RepositoryConnection, Session, User
from .services.analysis import PRAnalysisRequest, AnalysisService
from .services.workflow import DeliveryWorkflowService
from .schemas.models import WorkflowResponse
from .presentation import build_presentation
from .services.repair import RepairService
from .services.feedback import FeedbackService
from .services.stages import transition
from .services.ci_tools import redact

logger = logging.getLogger(__name__)

def failure_details(exc):
    # Format before logging so credentials in exception chains are also redacted.
    logger.error('Background analysis failed:\n%s', redact(traceback.format_exc()))
    return {'type': type(exc).__name__, 'message': redact(str(exc))[:2000]}

router = APIRouter(prefix='/api/product')
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='analysis')

def resume_incomplete_github_analyses():
    """Resume GitHub jobs interrupted by a local worker reload."""
    with DB() as db:
        stale_before=time.time()-300
        stale=db.scalars(select(Analysis).where(Analysis.status=='RUNNING',Analysis.updated_at<stale_before,Analysis.repository_connection_id.is_not(None),Analysis.pull_request_id.is_not(None))).all()
        for row in stale:
            if not (row.result or {}).get('finalized_demo'): row.status='PENDING'
        db.commit()
        rows=db.scalars(select(Analysis).where(Analysis.status=='PENDING',Analysis.repository_connection_id.is_not(None),Analysis.pull_request_id.is_not(None))).all()
        jobs=[(row.id,row.repository_connection_id,row.pull_request_id) for row in rows if not (row.result or {}).get('finalized_demo')]
    for analysis_id,connection_id,pull_request_id in jobs:
        executor.submit(run_github,analysis_id,connection_id,pull_request_id)
    return len(jobs)

def owned(db, analysis_id, user):
    row = db.get(Analysis, analysis_id)
    if not row or row.user_id != user.id: raise HTTPException(404, 'Analysis not found')
    return row

def event(analysis_id, kind, payload=None):
    with DB() as db:
        row = db.get(Analysis, analysis_id)
        if (row.result or {}).get('finalized_demo'): return
        row.events = [*row.events, {'id': len(row.events)+1, 'type': kind, 'timestamp': time.time(), 'data': payload or {}}]
        row.updated_at = time.time()
        row.result = {**(row.result or {}), 'stages': transition((row.result or {}).get('stages'), kind, payload)}
        if payload and 'graph' in payload:
            row.result = {**row.result, 'analysis': {**row.result.get('analysis', {}), 'graph': payload['graph']}}
        db.commit()

def run(analysis_id, request):
    try:
        with DB() as db:
            row = db.get(Analysis, analysis_id)
            if (row.result or {}).get('finalized_demo'): return
            row.status = 'RUNNING'; row.updated_at = time.time(); db.commit()
        with TemporaryDirectory(prefix='impact-sources-') as workspace:
            root = Path(workspace)
            for name, source in (request.sources or {}).items():
                target = root.joinpath(*PurePosixPath(name).parts)
                if not target.resolve().is_relative_to(root.resolve()): raise ValueError('Source path escapes workspace')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source, encoding='utf-8')
            scoped = request.model_copy(update={'repository_path': workspace})
            result = DeliveryWorkflowService().run(scoped, notify=lambda event_name, payload: event(analysis_id, event_name, payload))
        with DB() as db:
            row = db.get(Analysis, analysis_id); row.result = result.model_dump(mode='json'); row.status = 'COMPLETE'; row.updated_at = time.time()
            db.add(AuditEvent(actor_id=row.user_id, action='ANALYSIS_COMPLETE', subject_id=row.id)); db.commit()
        event(analysis_id, 'ANALYSIS_COMPLETE')
    except Exception as exc:
        internal = failure_details(exc)
        with DB() as db:
            row = db.get(Analysis, analysis_id); row.status = 'FAILED'; row.updated_at = time.time()
            row.result = {**(row.result or {}), 'error': {'code':'analysis_failed', **internal}}
            db.add(AuditEvent(actor_id=row.user_id, action='WORKFLOW_FAILED', subject_id=row.id)); db.commit()
        event(analysis_id, 'WORKFLOW_FAILED', internal)

def run_github(analysis_id: str, connection_id: str, pull_request_id: str):
    """Acquire one PR into a per-job checkout, then invoke the existing workflow."""
    try:
        # Both request handling and startup recovery may observe a newly queued
        # row. Only one worker may atomically claim it.
        with DB() as db:
            row = db.get(Analysis, analysis_id)
            if row and (row.result or {}).get('finalized_demo'): return
            claimed=db.execute(update(Analysis).where(Analysis.id==analysis_id,Analysis.status=='PENDING').values(status='RUNNING',updated_at=time.time()))
            db.commit()
            if claimed.rowcount != 1: return
        event(analysis_id, 'PR_ACQUISITION_STARTED')
        with DB() as db:
            row=db.get(Analysis,analysis_id); connection=db.get(RepositoryConnection,connection_id); pr=db.get(PullRequest,pull_request_id)
            if not row or not connection or not pr or row.user_id!=connection.user_id or pr.repository_connection_id!=connection.id: raise ValueError('Analysis ownership linkage is invalid')
            row.updated_at=time.time(); db.commit()
            installation_id=connection.installation_id; full_name=connection.full_name or connection.repository
            request=PRAnalysisRequest(repository=full_name,pull_request=pr.pr_number,base_sha=pr.base_sha,head_sha=pr.head_sha,title=pr.title)
        from . import github as github_routes
        live_pr=github_routes.service.get_pull_request(installation_id,full_name,request.pull_request)
        request=request.model_copy(update={'title':live_pr.get('title') or request.title,'body':live_pr.get('body'),'base_sha':live_pr['base']['sha'],'head_sha':live_pr['head']['sha']})
        with DB() as db:
            pr = db.get(PullRequest, pull_request_id)
            for key, value in github_routes.service.pull_request_values(live_pr).items():
                setattr(pr, key, value)
            row = db.get(Analysis, analysis_id)
            row.result = {**(row.result or {}), 'analysis': {'pr_metadata': request.model_dump(exclude={'sources', 'diff', 'repository_path'})}}
            db.commit()
        with TemporaryDirectory(prefix=f'impact-github-{analysis_id[:8]}-') as workspace:
            diff,sources,acquisition_evidence=github_routes.service.materialize_pull_request(installation_id,full_name,request.base_sha,request.head_sha,workspace)
            for item in acquisition_evidence:
                item=dict(item); kind=item.pop('type')
                event(analysis_id,kind,item)
            event(analysis_id, 'PR_DIFF_ACQUIRED', {'base_sha': request.base_sha, 'head_sha': request.head_sha, 'diff_bytes': len(diff.encode())})
            scoped=request.model_copy(update={'repository_path':workspace,'diff':diff,'sources':sources})
            result=DeliveryWorkflowService().run(scoped,validate=True,repository_path=workspace,notify=lambda event_name,payload:event(analysis_id,event_name,payload),workflow_id=analysis_id)
        with DB() as db:
            row=db.get(Analysis,analysis_id)
            result.stages.update((row.result or {}).get('stages') or {})
            row.result=result.model_dump(mode='json'); row.status='COMPLETE'; row.updated_at=time.time()
            db.add(AuditEvent(actor_id=row.user_id,action='GITHUB_PR_ANALYSIS_COMPLETE',subject_id=row.id,details={'repository_connection_id':connection_id,'pull_request_id':pull_request_id})); db.commit()
        event(analysis_id,'ANALYSIS_COMPLETE')
    except Exception as exc:
        internal = failure_details(exc)
        code=getattr(exc,'code','analysis_failed')
        messages={'checkout_failed':'Repository checkout failed.','diff_unavailable':'The pull request diff is unavailable.','unsupported_changes':'The pull request contains only unsupported or binary changes.','access_revoked':'GitHub repository access has been revoked.','rate_limited':'GitHub API rate limit reached.'}
        failed=False
        with DB() as db:
            row=db.get(Analysis,analysis_id)
            if row:
                message=messages.get(code,'Pull request analysis failed.')
                row.status='FAILED'; row.updated_at=time.time(); row.result={**(row.result or {}),'error':{'code':code,**internal}}; db.add(AuditEvent(actor_id=row.user_id,action='WORKFLOW_FAILED',subject_id=row.id,details={'code':code})); db.commit(); failed=True
        if failed: event(analysis_id,'WORKFLOW_FAILED',{'code':code,**internal})

def enqueue_github_analysis(pr_id: str, user_id: str):
    with DB() as db:
        pr=db.get(PullRequest,pr_id); connection=db.get(RepositoryConnection,pr.repository_connection_id) if pr else None
        if not pr or not connection or connection.user_id!=user_id or connection.connection_status!='CONNECTED': return None
        prior=db.scalars(select(Analysis).where(Analysis.user_id==user_id,Analysis.pull_request_id==pr.id).order_by(Analysis.created_at.desc())).all()
        for old in prior:
            metadata=((old.result or {}).get('analysis',old.result or {}).get('pr_metadata') or {})
            if old.status in ('PENDING','RUNNING') and metadata.get('head_sha')==pr.head_sha: return old.id
        row=Analysis(user_id=user_id,repository=connection.full_name or connection.repository,pull_request=pr.pr_number,repository_connection_id=connection.id,pull_request_id=pr.id)
        db.add(row); db.flush(); db.add(AuditEvent(actor_id=user_id,action='GITHUB_PR_ANALYSIS_CREATED',subject_id=row.id,details={'repository_connection_id':connection.id,'pull_request_id':pr.id})); db.commit(); analysis_id=row.id
    executor.submit(run_github,analysis_id,connection.id,pr.id)
    return analysis_id

def create_github_analysis(pr_id: str, user):
    with DB() as db:
        pr=db.get(PullRequest,pr_id); connection=db.get(RepositoryConnection,pr.repository_connection_id) if pr else None
        if not pr or not connection or connection.user_id!=user.id: raise HTTPException(404,'Pull request not found')
        if connection.connection_status!='CONNECTED': raise HTTPException(409,'GitHub repository access is not connected')
        active=db.scalar(select(Analysis).where(Analysis.user_id==user.id,Analysis.status.in_(['PENDING','RUNNING'])))
        if active: raise HTTPException(409,'An analysis is already running')
    analysis_id=enqueue_github_analysis(pr_id,user.id)
    if not analysis_id: raise HTTPException(409,'Analysis could not be queued')
    return {'id':analysis_id,'status':'PENDING'}

@router.post('/analyses', status_code=202)
def create(body: PRAnalysisRequest, user=Depends(current_user)):
    # Browser accounts submit source text, never arbitrary server filesystem paths.
    if body.repository_path: raise HTTPException(422, 'Submit a diff and source files; server paths are not accepted by browser accounts')
    if not body.diff or len(body.diff) > 2_000_000 or sum(len(v) for v in (body.sources or {}).values()) > 5_000_000:
        raise HTTPException(422, 'A diff is required (2 MB maximum; sources 5 MB maximum)')
    for name in (body.sources or {}):
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or chr(92) in name or ':' in name or not path.parts or any(part.startswith('.') for part in path.parts):
            raise HTTPException(422, 'Source paths must be relative project files')
    with DB() as db:
        active = db.scalar(select(Analysis).where(Analysis.user_id == user.id, Analysis.status.in_(['PENDING','RUNNING'])))
        if active: raise HTTPException(409, 'An analysis is already running')
        row = Analysis(user_id=user.id, repository=body.repository, pull_request=body.pull_request)
        db.add(row); db.flush()
        if not db.scalar(select(RepositoryConnection).where(RepositoryConnection.user_id == user.id, RepositoryConnection.repository == body.repository)):
            db.add(RepositoryConnection(user_id=user.id, repository=body.repository))
        db.add(AuditEvent(actor_id=user.id, action='ANALYSIS_CREATED', subject_id=row.id, details={'repository': body.repository, 'pull_request': body.pull_request}))
        db.commit(); analysis_id = row.id
    executor.submit(run, analysis_id, body)
    return {'id': analysis_id, 'status': 'PENDING'}

def summary(row):
    result = row.result or {}
    result = result.get('analysis', result)
    workflow = row.result or {}
    risk = result.get('risk_prediction') or {}
    validation = workflow.get('validation') or result.get('validation') or {}
    deployment = workflow.get('deployment') or result.get('deployment') or {}
    probability = risk.get('risk_probability')
    threshold = risk.get('threshold')
    return {'id': row.id, 'repository': row.repository, 'pull_request': row.pull_request, 'repository_connection_id':row.repository_connection_id,'pull_request_id':row.pull_request_id,'created_at': row.created_at, 'updated_at': row.updated_at or row.created_at, 'status': row.status, 'decision': (workflow.get('decision') or result.get('decision') or {}).get('decision'), 'risk': probability, 'risk_label': ('HIGH' if probability >= (threshold if threshold is not None else .5) else 'LOW') if probability is not None else 'UNAVAILABLE', 'validation':validation.get('status'), 'approval':workflow.get('approval_state'), 'deployment':deployment.get('lifecycle_state') or deployment.get('status'), 'error':workflow.get('error')}

@router.get('/overview')
def overview(user=Depends(current_user)):
    with DB() as db:
        rows=db.scalars(select(Analysis).where(Analysis.user_id==user.id).order_by(Analysis.updated_at.desc())).all()
        connections=db.scalars(select(RepositoryConnection).where(RepositoryConnection.user_id==user.id,RepositoryConnection.provider=='github',RepositoryConnection.connection_status=='CONNECTED')).all()
        ids=[r.id for r in connections]
        open_prs=db.scalar(select(func.count()).select_from(PullRequest).where(PullRequest.repository_connection_id.in_(ids or ['']),PullRequest.state=='open')) or 0
        def payload(row): return (row.result or {}).get('analysis',row.result or {})
        return {'connected_repositories':len(connections),'open_pull_requests':open_prs,'analyses_running':sum(a.status in ('PENDING','RUNNING') for a in rows),'completed_analyses':sum(a.status=='COMPLETE' for a in rows),'high_risk_changes':sum(((payload(a).get('risk_prediction') or {}).get('risk_probability') or 0)>=.5 for a in rows),'awaiting_approval':sum((a.result or {}).get('approval_state')=='READY_FOR_APPROVAL' for a in rows),'recent_activity':[summary(a) for a in rows[:8]]}

@router.get('/analyses')
def analyses(user=Depends(current_user)):
    with DB() as db:
        return {'items': [summary(row) for row in db.scalars(select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).limit(100))]}

@router.get('/analyses/{analysis_id}')
def detail(analysis_id: str, user=Depends(current_user)):
    with DB() as db:
        row = owned(db, analysis_id, user)
        stored = row.result or {}
        live={**stored,'provider_readiness':RepairService().readiness()} if stored.get('workflow_id') else stored
        return {**summary(row), 'result': stored.get('analysis', stored), 'workflow': live if row.status == 'COMPLETE' and stored.get('workflow_id') else None, 'events': row.events, 'presentation': build_presentation(row.id, row.status, live)}

@router.get('/analyses/{analysis_id}/events')
def stream(analysis_id: str, request: Request, after: int = 0, user=Depends(current_user)):
    with DB() as db: owned(db, analysis_id, user)
    try: cursor = max(after, int(request.headers.get('last-event-id', '0')))
    except ValueError: cursor = after
    session_hash = digest(request.cookies.get(COOKIE, ''))
    async def events():
        nonlocal cursor
        while not await request.is_disconnected():
            with DB() as db:
                session = db.get(Session, session_hash); account = db.get(User, user.id)
                if not session or session.expires_at <= time.time() or not account or not account.is_active: return
                row = owned(db, analysis_id, user)
                pending = [e for e in row.events if e['id'] > cursor]
                status = row.status
            for item in pending:
                cursor = item['id']
                yield f"id: {cursor}\ndata: {json.dumps(item)}\n\n"
            if status in ('COMPLETE','FAILED') and (not pending or pending[-1]['type'] in ('ANALYSIS_COMPLETE','WORKFLOW_FAILED')): return
            yield ': keepalive\n\n'
            await asyncio.sleep(1)
    return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})


class ApprovalBody(BaseModel):
    note: str | None = Field(default=None, max_length=1000)

class SelectionBody(BaseModel):
    variant: str
    reason: str | None = Field(default=None, max_length=1000)

class LearningBody(BaseModel):
    deployment_outcome: str
    prediction_usefulness: str
    repair_usefulness: str
    note: str | None = Field(default=None, max_length=2000)

def _persist_workflow(row, result, actor, action, data=None):
    if (row.result or {}).get('finalized_demo'):
        if action not in {'VARIANT_SELECTED', 'FEEDBACK_RECORDED', 'WORKFLOW_APPROVED', 'WORKFLOW_REJECTED'}:
            raise HTTPException(409, 'Final demo evidence is immutable')
        # Only explicit human selection, approval and feedback may evolve.
        preserved = dict(row.result)
        proposed = result.model_dump(mode='json')
        for key in ('selection', 'approval_state', 'approval_metadata', 'deployment', 'feedback', 'monitoring', 'stages'):
            preserved[key] = proposed[key]
        result = WorkflowResponse.model_validate(preserved)
    occurred=time.time(); row.result=result.model_dump(mode='json'); row.updated_at=occurred
    row.events=[*row.events,{'id':len(row.events)+1,'type':action,'timestamp':occurred,'data':data or {}}]
    return occurred

def _materialize_public_github(full_name: str, base_sha: str, head_sha: str, workspace: str) -> tuple[str, dict, list]:
    """Credential-free fallback for an explicitly public GitHub repository."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",full_name): raise ValueError('Invalid GitHub repository name')
    url=f"https://github.com/{full_name}.git"
    cloned=subprocess.run(['git','clone','--quiet','--no-checkout',url,workspace],capture_output=True,text=True,timeout=120,shell=False,check=False)
    if cloned.returncode: raise ValueError('Public repository checkout is unavailable')
    checked=subprocess.run(['git','checkout','--quiet',head_sha],cwd=workspace,capture_output=True,text=True,timeout=60,shell=False,check=False)
    if checked.returncode: raise ValueError('Persisted PR head commit is unavailable')
    diff=subprocess.run(['git','diff','--no-ext-diff','--binary',base_sha,head_sha],cwd=workspace,capture_output=True,text=True,timeout=60,shell=False,check=False)
    if diff.returncode or not diff.stdout: raise ValueError('Persisted PR diff is unavailable')
    return diff.stdout,{},[{'type':'PUBLIC_GITHUB_REACQUIRED','repository':full_name,'head_sha':head_sha}]

@router.post('/analyses/{analysis_id}/repair')
def generate_repair(analysis_id: str, user=Depends(current_user)):
    with DB() as db:
        row=owned(db,analysis_id,user)
        if row.status!='COMPLETE' or not (row.result or {}).get('workflow_id'): raise HTTPException(409,'A completed workflow is required')
        stored=row.result; connection_id=row.repository_connection_id; pr_id=row.pull_request_id
        if stored.get('finalized_demo'):
            return {'workflow':stored,'reused_persisted_candidate':True,'finalized_demo':True}
        candidate=((stored.get('repair') or {}).get('candidate') or {})
        if candidate.get('status')=='validated' and candidate.get('selectable') is True:
            return {'workflow':stored,'reused_persisted_candidate':True}
    readiness=RepairService().readiness()
    if readiness['status']!='READY': raise HTTPException(409,'The live LLM provider is not ready')
    if not connection_id or not pr_id: raise HTTPException(409,'Candidate generation requires the persisted GitHub source acquisition linkage')
    with DB() as db:
        connection=db.get(RepositoryConnection,connection_id); pr=db.get(PullRequest,pr_id)
        if not connection or not pr: raise HTTPException(409,'GitHub source linkage is unavailable')
        installation_id=connection.installation_id; full_name=connection.full_name or connection.repository
    from . import github as github_routes
    metadata=stored['analysis']['pr_metadata']
    base_sha,head_sha=metadata.get('base_sha'),metadata.get('head_sha')
    if not base_sha or not head_sha: raise HTTPException(409,'Persisted analyzed commit identities are missing; re-analyze this PR')
    try:
        with TemporaryDirectory(prefix=f'repair-{analysis_id[:8]}-') as workspace:
            try:
                diff,_,_=github_routes.service.materialize_pull_request(installation_id,full_name,base_sha,head_sha,workspace)
            except Exception as acquisition_error:
                if getattr(acquisition_error,'code',None)!='not_configured': raise
                diff,_,_=_materialize_public_github(full_name,base_sha,head_sha,workspace)
            service=DeliveryWorkflowService(); result=WorkflowResponse.model_validate(stored)
            result=service.generate_candidate(result,root=workspace,diff=diff)
    except Exception as exc:
        code=getattr(exc,'code',type(exc).__name__)
        raise HTTPException(502,f'Repair proposal could not be generated safely: {code}') from exc
    with DB() as db:
        row=owned(db,analysis_id,user); _persist_workflow(row,result,user,'REPAIR_PROPOSAL_GENERATED',{'status':result.repair.status,'candidate':result.repair.candidate.get('status')})
        db.add(AuditEvent(actor_id=user.id,action='REPAIR_PROPOSAL_GENERATED',subject_id=row.id,details={'status':result.repair.status})); db.commit()
    return {'workflow':result.model_dump(mode='json')}

@router.post('/analyses/{analysis_id}/selection')
def select_variant(analysis_id: str, body: SelectionBody, user=Depends(current_user)):
    variant=body.variant.upper()
    if variant not in ('ORIGINAL','AI_CANDIDATE'): raise HTTPException(422,'Choose ORIGINAL or AI_CANDIDATE')
    with DB() as db:
        row=owned(db,analysis_id,user); result=WorkflowResponse.model_validate(row.result or {})
        if variant=='AI_CANDIDATE' and (result.repair.candidate.get('status')!='validated' or not result.repair.candidate.get('selectable')): raise HTTPException(409,'No validated AI candidate is available')
        occurred=time.time(); result.selection={'variant':variant,'actor_id':user.id,'actor':user.display_name,'timestamp':occurred,'reason':body.reason}
        if result.deployment:
            result.deployment=result.deployment.model_copy(update={'lifecycle_state':'READY_FOR_APPROVAL','simulation_status':'BLOCKED','simulation_details':{'mode':'simulation','production_changed':False,'selected_variant':variant,'message':'The selected variant requires explicit approval before simulation.'}})
        store=FeedbackService(); previous=store.get(result.workflow_id)
        if previous:
            values=previous.model_dump(); values['selected_variant']=variant
            values['metadata']={**(previous.metadata or {}),'variant_selection':{'variant':variant,'actor_id':user.id,'actor':user.display_name,'timestamp':occurred,'note':body.reason}}
            store.record(**values)
        result.approval_state='READY_FOR_APPROVAL'; _persist_workflow(row,result,user,'VARIANT_SELECTED',{'variant':variant,'reason':body.reason})
        db.add(AuditEvent(actor_id=user.id,action='VARIANT_SELECTED',subject_id=row.id,details={'variant':variant,'reason':body.reason})); db.commit()
        return {'workflow':row.result}

@router.post('/analyses/{analysis_id}/feedback')
def learning_feedback(analysis_id: str, body: LearningBody, user=Depends(current_user)):
    aliases={'SUCCESSFUL':'SUCCESS','ROLLED_BACK':'WOULD_ROLLBACK'}
    allowed={'deployment_outcome':{'SUCCESS','ISSUE_DETECTED','WOULD_ROLLBACK'},'prediction_usefulness':{'ACCURATE','TOO_CAUTIOUS','TOO_OPTIMISTIC'},'repair_usefulness':{'HELPFUL','NOT_HELPFUL','NOT_USED'}}
    values={k:aliases.get(getattr(body,k).upper(),getattr(body,k).upper()) for k in allowed}
    if any(values[k] not in allowed[k] for k in allowed): raise HTTPException(422,'Invalid feedback choice')
    with DB() as db:
        row=owned(db,analysis_id,user); result=WorkflowResponse.model_validate(row.result or {})
        store=FeedbackService(); previous=store.get(result.workflow_id); payload=previous.model_dump() if previous else {'workflow_id':result.workflow_id,'repository':row.repository,'pull_request':row.pull_request}
        payload.update(values); payload['user_note']=body.note; payload['selected_variant']=(result.selection or {}).get('variant'); payload['approval']=(result.approval_metadata or {}).get('decision'); payload['outcome_known']=True
        payload['metadata']={**(payload.get('metadata') or {}),'analysis_id':analysis_id,'repair_attempts':result.repair_attempts,'candidate_produced':bool(result.repair.patch),'candidate_status':result.repair.candidate.get('status'),'test_generation':result.test_generation.model_dump(mode='json'),'post_candidate_decision':result.post_candidate_decision.decision if result.post_candidate_decision else None,'security_result':((result.repair.candidate.get('validation') or {}).get('checks') or {}).get('security'),'test_result':((result.repair.candidate.get('validation') or {}).get('checks') or {}).get('regression')}
        record=store.record(**payload); result.feedback={'status':'RECORDED','timestamp':time.time(),'deployment_outcome':values['deployment_outcome'],'prediction_usefulness':values['prediction_usefulness'],'repair_usefulness':values['repair_usefulness'],'note':body.note,'model_automatically_modified':False}
        result.stages.setdefault('FEEDBACK',{}).update({'status':'completed','summary':'Stored for future evaluation; production model unchanged'})
        _persist_workflow(row,result,user,'FEEDBACK_RECORDED',values); db.add(AuditEvent(actor_id=user.id,action='FEEDBACK_RECORDED',subject_id=row.id,details=values)); db.commit()
        return {'feedback':result.feedback,'record':record.model_dump(mode='json')}

@router.post('/analyses/{analysis_id}/{action}')
def approval(analysis_id: str, action: str, body: ApprovalBody | None = None, user=Depends(current_user)):
    if action not in ('approve', 'reject'): raise HTTPException(404, 'Action not found')
    with DB() as db:
        row = db.scalar(select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id).with_for_update())
        if not row: raise HTTPException(404, 'Analysis not found')
        if row.status != 'COMPLETE' or not row.result or 'analysis' not in row.result: raise HTTPException(409, 'No workflow available')
        result = WorkflowResponse.model_validate(row.result)
        if action=='approve' and (result.selection or {}).get('variant') not in ('ORIGINAL','AI_CANDIDATE'): raise HTTPException(409,'Select ORIGINAL or AI_CANDIDATE before approval')
        service = DeliveryWorkflowService()
        service.workflows[result.workflow_id] = result
        try: result = service.approve(result.workflow_id) if action == 'approve' else service.reject(result.workflow_id)
        except ValueError as exc: raise HTTPException(409, str(exc))
        occurred = time.time()
        result.approval_metadata = {'actor_id':user.id,'actor':user.display_name,'decision':'APPROVED' if action=='approve' else 'REJECTED','timestamp':occurred,'note':body.note if body else None}
        row.result = result.model_dump(mode='json')
        row.updated_at = time.time()
        new_events = [('USER_APPROVED', {'note':body.note if body else None}) , ('DEPLOYMENT_STARTED', {'mode':'simulation'}), ('DEPLOYMENT_COMPLETED', {'mode':'simulation','production_changed':False})] if action == 'approve' else [('USER_REJECTED', {'note':body.note if body else None})]
        events=list(row.events)
        for kind,data in new_events: events.append({'id':len(events)+1,'type':kind,'timestamp':occurred,'data':data})
        row.events=events
        db.add(AuditEvent(actor_id=user.id, action='USER_APPROVED' if action == 'approve' else 'USER_REJECTED', subject_id=row.id, details={'note':body.note if body else None,'mode':'simulation'})); db.commit()
        return {'workflow': row.result}
