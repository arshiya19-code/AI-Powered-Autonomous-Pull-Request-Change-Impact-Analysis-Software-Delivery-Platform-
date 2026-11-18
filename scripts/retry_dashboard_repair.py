"""Retry live repair on a freshly analyzed, immutable GitHub snapshot."""
import argparse
import copy
import json
from pathlib import Path
from backend.app.database import DB, Analysis, User
from backend.app.product import generate_repair
from backend.app.services.ci_tools import redact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_id')
    parser.add_argument('--output', required=True)
    parser.add_argument('--invalidate-test-reason', help='Record a discovered test-evidence defect and revalidate the preserved production patch')
    args = parser.parse_args()
    with DB() as db:
        row = db.get(Analysis, args.analysis_id)
        if row is None or row.status != 'COMPLETE':
            raise ValueError('A completed persisted analysis is required')
        if (row.result or {}).get('finalized_demo'):
            raise ValueError('Frozen demo replay is not a live verification')
        if args.invalidate_test_reason:
            evidence = copy.deepcopy(row.result)
            candidate = evidence['repair']['candidate']
            assert candidate.get('persisted_source') and candidate.get('patch')
            candidate.update(selectable=False, status='rejected', reason=args.invalidate_test_reason)
            if evidence.get('repair_attempts'):
                evidence['repair_attempts'][-1]['failure_feedback'] = {'candidate_error': args.invalidate_test_reason}
            row.result = evidence
            db.commit()
        if ((row.result or {}).get('repair') or {}).get('candidate', {}).get('selectable'):
            raise ValueError('Existing selectable candidates must not be reported as fresh AI runs')
        user = db.get(User, row.user_id)
    error = None
    try:
        generate_repair(args.analysis_id, user=user)
    except Exception as exc:
        error = redact(str(getattr(exc, 'detail', exc)))
    with DB() as db:
        row = db.get(Analysis, args.analysis_id)
        item = {'repository':row.repository, 'number':row.pull_request,
                'pr_id':row.pull_request_id, 'analysis_id':row.id, 'status':row.status,
                'repair_error':error, 'result':row.result or {}}
    Path(args.output).write_text(json.dumps([item], indent=2), encoding='utf-8')
    candidate = (item['result'].get('repair') or {}).get('candidate') or {}
    print(json.dumps({'analysis_id':row.id, 'error':error, 'candidate':candidate.get('status'),
                      'selectable':candidate.get('selectable'),
                      'attempts':[(a.get('status'),a.get('failure_reason')) for a in item['result'].get('repair_attempts',[])]}), flush=True)


if __name__ == '__main__':
    main()
