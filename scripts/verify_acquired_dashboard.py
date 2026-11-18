"""Reverify actual dashboard commits using already-acquired Git snapshots and live AI."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from sqlalchemy import select
from backend.app.database import DB,Analysis,PullRequest,RepositoryConnection
from backend.app.services.analysis import PRAnalysisRequest
from backend.app.services.workflow import DeliveryWorkflowService
from backend.app.services.ci_tools import redact
from backend.app.presentation import build_presentation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pr-id')
    args=parser.parse_args()
    output=Path('reports/dashboard_verified.json')
    results=json.loads(output.read_text(encoding='utf-8')) if output.exists() else []
    with DB() as db:
        prs=db.scalars(select(PullRequest).join(RepositoryConnection).where(RepositoryConnection.provider=='github')).all()
    for pr in prs:
        if args.pr_id and pr.id!=args.pr_id: continue
        with DB() as db:
            conn=db.get(RepositoryConnection,pr.repository_connection_id)
            row=Analysis(user_id=conn.user_id,repository=conn.full_name,pull_request=pr.pr_number,repository_connection_id=conn.id,pull_request_id=pr.id,status='RUNNING')
            db.add(row);db.commit();aid=row.id
            previous=db.scalar(select(Analysis).where(Analysis.pull_request_id==pr.id,Analysis.status=='COMPLETE').order_by(Analysis.created_at.desc()))
            meta=((previous.result or {}).get('analysis') or {}).get('pr_metadata') or {} if previous else {}
        print(json.dumps({'phase':'start','repository':conn.full_name,'number':pr.pr_number,'analysis_id':aid}),flush=True)
        result={}
        try:
            with TemporaryDirectory(prefix='dashboard-verify-') as workspace:
                source=Path('.tmp/dashboard-sources')/pr.id
                def git(*args):
                    p=subprocess.run(['git',*args],cwd=workspace,capture_output=True,text=True,timeout=60)
                    if p.returncode: raise RuntimeError(redact(p.stderr))
                    return p.stdout
                git('clone','--quiet','--no-local',str(source.resolve()),'.')
                # The acquired repositories keep both exact commits; copy refs
                # explicitly because their checkout is intentionally detached.
                git('fetch','--quiet',str(source.resolve()),pr.base_sha,pr.head_sha)
                git('checkout','--quiet','--detach',pr.head_sha)
                diff=git('diff','--no-ext-diff','--binary',pr.base_sha,pr.head_sha)
                request=PRAnalysisRequest(repository=conn.full_name,pull_request=pr.pr_number,title=pr.title,body=meta.get('body'),base_sha=pr.base_sha,head_sha=pr.head_sha,repository_path=workspace)
                service=DeliveryWorkflowService()
                flow=service.run(request,validate=True,repository_path=workspace,workflow_id=aid)
                print(json.dumps({'phase':'analyzed','analysis_id':aid,'diff_bytes':len(diff.encode()),'validation':flow.validation.status}),flush=True)
                with DB() as db:
                    row=db.get(Analysis,aid);row.result=flow.model_dump(mode='json');row.status='COMPLETE';db.commit()
                if diff.strip():
                    flow=service.generate_candidate(flow,root=workspace,diff=diff)
                else:
                    flow.repair.issue_summary='EMPTY_DIFF: base and head have identical tracked content; no PR defect can be attributed to this change.'
                result=flow.model_dump(mode='json')
                status='COMPLETE'
        except Exception as exc:
            status='FAILED';result={'error':{'type':type(exc).__name__,'message':redact(str(exc))[:2000]}}
        with DB() as db:
            row=db.get(Analysis,aid);row.result=result;row.status=status;row.updated_at=time.time();db.commit()
        item={'repository':conn.full_name,'number':pr.pr_number,'pr_id':pr.id,'analysis_id':aid,'status':status,'result':result,'presentation':build_presentation(aid,status,result)}
        results=[x for x in results if x['pr_id']!=pr.id]+[item]
        output.write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps({'phase':'finished','analysis_id':aid,'status':status,'candidate':((result.get('repair') or {}).get('candidate') or {}).get('status'),'attempts':[(a['status'],a.get('failure_reason')) for a in result.get('repair_attempts',[])]}),flush=True)

if __name__=='__main__': main()
