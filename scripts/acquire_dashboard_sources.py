from pathlib import Path
from sqlalchemy import select
from backend.app.database import DB,PullRequest,RepositoryConnection
from backend.app.services.github import GitHubService
s=GitHubService()
with DB() as db:
 for pr in db.scalars(select(PullRequest).join(RepositoryConnection).where(RepositoryConnection.provider=='github')):
  c=db.get(RepositoryConnection,pr.repository_connection_id)
  root=Path('.tmp/dashboard-sources')/pr.id
  if not (root/'.git').exists():
   diff,sources,e=s.materialize_pull_request(c.installation_id,c.full_name,pr.base_sha,pr.head_sha,root)
   (root.parent/(pr.id+'.diff')).write_text(diff,encoding='utf-8')
  print(c.full_name,pr.pr_number,str(root),flush=True)
