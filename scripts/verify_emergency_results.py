"""Report newest dashboard evidence without creating, repairing or replaying candidates."""
import argparse
import json
from pathlib import Path

from sqlalchemy import select

from backend.app.database import Analysis, DB, PullRequest, RepositoryConnection
from backend.app.services.github import GitHubService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', default='reports/emergency_final_summary.json')
    args = parser.parse_args()
    service = GitHubService() if args.live else None
    rows = []
    with DB() as db:
        prs = db.execute(select(PullRequest, RepositoryConnection).join(RepositoryConnection)
                         .where(RepositoryConnection.provider == 'github')).all()
        for pr, connection in prs:
            analysis = db.scalar(select(Analysis).where(Analysis.pull_request_id == pr.id)
                                 .order_by(Analysis.created_at.desc()))
            result = analysis.result or {} if analysis else {}
            repair = result.get('repair') or {}
            candidate = repair.get('candidate') or {}
            metadata = (result.get('analysis') or {}).get('pr_metadata') or {}
            checks = (candidate.get('validation') or {}).get('checks') or {}
            generated = result.get('test_generation') or {}
            reanalysis = candidate.get('reanalysis') or {}
            live = service.get_pull_request(connection.installation_id, connection.full_name, pr.pr_number) if service else None
            current_commits = (not live or (metadata.get('base_sha') == live['base']['sha']
                                           and metadata.get('head_sha') == live['head']['sha']))
            item = {'repository': connection.full_name, 'pr': pr.pr_number,
                    'analysis_id': analysis.id if analysis else None, 'analysis_status': analysis.status if analysis else None,
                    'provider': repair.get('provider'), 'model': repair.get('model'),
                    'explanation': repair.get('explanation'), 'base': metadata.get('base_sha'), 'head': metadata.get('head_sha'),
                    'current_commits': current_commits, 'fresh': not result.get('finalized_demo') and not result.get('replayed'),
                    'patch_valid': bool(candidate.get('patch') and candidate.get('persisted_source')),
                    'checks': {name: check.get('status') for name, check in checks.items()},
                    'regression_generated': bool(generated.get('generated_tests')),
                    'integration_paths': [test.get('dependency_path') for test in generated.get('generated_integration_tests', [])],
                    'reanalysis': reanalysis.get('status'), 'feature_count': reanalysis.get('feature_count'),
                    'comparison_available': bool(result.get('candidate_comparison')),
                    'selectable': candidate.get('selectable') is True}
            item['verified'] = all([item['selectable'], item['fresh'], current_commits, item['patch_valid'],
                                    item['regression_generated'], item['comparison_available'], item['reanalysis'] == 'AVAILABLE',
                                    item['feature_count'] == 42,
                                    all(item['checks'].get(key) == 'PASSED' for key in
                                        ('build', 'unit', 'regression', 'lint', 'security', 'dependency')),
                                    item['checks'].get('targeted_integration') == 'PASSED' if item['integration_paths'] else True])
            rows.append(item)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)
    return 0 if rows and all(row['verified'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
