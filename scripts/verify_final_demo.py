"""Read and certify the finalized evidence; never choose a variant or call AI."""
import hashlib
import json
from pathlib import Path

from backend.app.database import Analysis, DB, User, engine
from backend.app.product import generate_repair
from backend.app.schemas.models import WorkflowResponse

ANALYSIS_ID = '5a77faad-eca8-42f6-ad30-a545d9267e90'


def main():
    with DB() as db:
        row = db.get(Analysis, ANALYSIS_ID)
        result = row.result
        user = db.get(User, row.user_id)
    assert Path(engine.url.database).resolve() == Path('backend/accounts.sqlite3').resolve()
    assert result['finalized_demo'] and result['replayed']
    assert result['provider_request_count'] == 0
    assert not result['selection'].get('variant')
    candidate = result['repair']['candidate']
    assert candidate['status'] == 'validated' and candidate['selectable']
    assert candidate['reanalysis']['status'] == 'AVAILABLE'
    checks = candidate['validation']['checks']
    for key in ('build', 'unit', 'regression', 'targeted_integration', 'lint', 'security', 'dependency'):
        assert checks[key]['status'] == 'PASSED', key
    for file in candidate['files']:
        assert all(file.get(key) for key in ('file', 'original_code', 'candidate_code', 'unified_diff'))
    assert result['candidate_comparison']['recommendation'] == 'USE_CANDIDATE'
    roundtrip = WorkflowResponse.model_validate(result).model_dump(mode='json')
    for key in ('repair', 'analysis', 'validation', 'test_generation', 'candidate_comparison'):
        assert roundtrip[key] == result[key], f'Schema changed frozen {key}'
    reused = generate_repair(ANALYSIS_ID, user=user)
    assert reused['reused_persisted_candidate'] and reused['workflow'] == result
    serialized = json.dumps(result, sort_keys=True, indent=2)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    path = Path('reports/final_demo_snapshot.json')
    if path.exists():
        assert path.read_text(encoding='utf-8') == serialized, 'Final evidence changed after restart'
    else:
        path.write_text(serialized, encoding='utf-8')
    print(json.dumps({'analysis_id': ANALYSIS_ID, 'candidate': candidate['status'],
                      'selectable': candidate['selectable'], 'selected_variant': 'NONE',
                      'provider_request_count': 0, 'checks': {k: v['status'] for k, v in checks.items()},
                      'snapshot_sha256': digest, 'repair_reuses_immutable_snapshot': True}, indent=2))


if __name__ == '__main__':
    main()
