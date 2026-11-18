"""Run fresh persisted workflows for every GitHub dashboard PR, without replay."""
import argparse
import json
from pathlib import Path
from sqlalchemy import select
from backend.app.database import DB, Analysis, PullRequest, RepositoryConnection, User
from backend.app.product import run_github, generate_repair
from backend.app.services.ci_tools import redact

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr-id')
    parser.add_argument('--first-pr-id')
    parser.add_argument('--skip-pr-id', action='append', default=[])
    parser.add_argument('--output', default='reports/dashboard_live_reproduction.json')
    args = parser.parse_args()
    with DB() as db:
        prs=db.scalars(select(PullRequest).join(RepositoryConnection).where(RepositoryConnection.provider=='github')).all()
    prs.sort(key=lambda pr: (pr.id != args.first_pr_id, pr.pr_number))
    prs = [pr for pr in prs if pr.id not in args.skip_pr_id]
    if args.pr_id:
        prs = [pr for pr in prs if pr.id == args.pr_id]
    outcomes=[]
    for pr in prs:
        with DB() as db:
            conn=db.get(RepositoryConnection,pr.repository_connection_id)
            user=db.get(User,conn.user_id)
            row=Analysis(user_id=user.id,repository=conn.full_name or conn.repository,pull_request=pr.pr_number,repository_connection_id=conn.id,pull_request_id=pr.id)
            db.add(row);db.commit(); aid=row.id
        print(json.dumps({'phase':'start','repository':row.repository,'number':pr.pr_number,'analysis_id':aid}),flush=True)
        run_github(aid,conn.id,pr.id)
        with DB() as db:
            row=db.get(Analysis,aid)
            status=row.status
        print(json.dumps({'phase':'analyzed','analysis_id':aid,'status':status}),flush=True)
        repair_error=None
        if status=='COMPLETE':
            try: generate_repair(aid,user=user)
            except Exception as exc: repair_error=redact(str(getattr(exc,'detail',exc)))
        with DB() as db:
            row=db.get(Analysis,aid);result=row.result or {}
        item={'repository':row.repository,'number':pr.pr_number,'pr_id':pr.id,'analysis_id':aid,'status':row.status,'repair_error':repair_error,'result':result}
        outcomes.append(item)
        Path(args.output).write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
        candidate = (result.get('repair') or {}).get('candidate') or {}
        print(json.dumps({**{k:v for k,v in item.items() if k!='result'}, 'candidate_status':candidate.get('status'), 'selectable':candidate.get('selectable'), 'attempts':[(a.get('status'),a.get('failure_reason')) for a in result.get('repair_attempts',[])]}),flush=True)
if __name__=='__main__': main()
