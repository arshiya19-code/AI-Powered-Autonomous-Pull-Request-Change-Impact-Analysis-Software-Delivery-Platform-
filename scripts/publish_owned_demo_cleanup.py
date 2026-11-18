"""Publish reviewed tracked-file cleanup to an owned PR head, with a concurrency check."""
import argparse
import json
import subprocess
from pathlib import Path

from sqlalchemy import select

from backend.app.database import DB, RepositoryConnection
from backend.app.services.github import GitHubService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--message', required=True)
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    meta = json.loads(root.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    repo = meta['repository']
    assert repo.startswith('arshiya19-code/')
    with DB() as db:
        connection = db.scalar(select(RepositoryConnection).where(RepositoryConnection.full_name == repo))
    service = GitHubService()
    live = service.get_pull_request(connection.installation_id, repo, meta['number'])
    assert live['head']['repo']['full_name'] == repo
    assert live['head']['ref'] == meta['branch'] != live['base']['repo']['default_branch']
    assert live['head']['sha'] == meta['head'], 'PR changed since inspection; stop rather than overwrite'
    paths = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=M'], cwd=root, text=True).splitlines()
    assert paths and all(not path.startswith('tests/ai_generated/') for path in paths)
    headers = {'Authorization': 'Bearer ' + service.installation_token(connection.installation_id),
               'Accept': 'application/vnd.github+json'}

    def request(method, path, data=None):
        response = service.client.request(method, service.api + f'/repos/{repo}' + path,
                                          headers=headers, json=data)
        return service._response(response)

    parent = request('GET', '/git/commits/' + meta['head'])
    entries = [{'path': path, 'mode': '100644', 'type': 'blob',
                'content': (root / path).read_text(encoding='utf-8')} for path in paths]
    tree = request('POST', '/git/trees', {'base_tree': parent['tree']['sha'], 'tree': entries})
    commit = request('POST', '/git/commits', {'message': args.message, 'tree': tree['sha'], 'parents': [meta['head']]})
    request('PATCH', '/git/refs/heads/' + meta['branch'], {'sha': commit['sha'], 'force': False})
    evidence = {**meta, 'new_head': commit['sha'], 'modified_files': paths, 'message': args.message}
    root.with_suffix('.published.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
