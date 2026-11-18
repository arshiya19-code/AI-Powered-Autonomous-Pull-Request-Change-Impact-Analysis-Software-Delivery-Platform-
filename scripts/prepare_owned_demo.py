"""Inspect and prepare an owned demo branch; never creates AI candidate patches."""
import argparse
import json
from pathlib import Path

from backend.app.database import DB, PullRequest, RepositoryConnection
from backend.app.services.github import GitHubService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pr_id')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with DB() as db:
        pr = db.get(PullRequest, args.pr_id)
        connection = db.get(RepositoryConnection, pr.repository_connection_id)
    assert connection.full_name.startswith('arshiya19-code/')
    service = GitHubService()
    live = service.get_pull_request(connection.installation_id, connection.full_name, pr.pr_number)
    assert live['head']['repo']['full_name'] == connection.full_name
    assert live['head']['ref'] != live['base']['repo']['default_branch']
    root = Path(args.output).resolve()
    assert not root.exists(), 'Use a fresh inspection directory'
    diff, _, _ = service.materialize_pull_request(connection.installation_id, connection.full_name,
                                                 live['base']['sha'], live['head']['sha'], root)
    metadata = {'repository': connection.full_name, 'number': pr.pr_number,
                'branch': live['head']['ref'], 'base': live['base']['sha'], 'head': live['head']['sha'],
                'title': live['title'], 'body': live.get('body'), 'changed_files': live['changed_files']}
    root.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    root.with_suffix('.diff').write_text(diff, encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
