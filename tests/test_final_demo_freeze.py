"""A finalized demo must survive stale writers while allowing explicit selection."""
import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from backend.app import database, product
from backend.app.schemas.models import WorkflowResponse


def test_database_rejects_stale_evidence_and_flag_writes(tmp_path, monkeypatch):
    engine = create_engine('sqlite:///' + str(tmp_path / 'freeze.sqlite3'))
    monkeypatch.setattr(database, 'engine', engine)
    monkeypatch.setattr(database, 'url', str(engine.url))
    database.initialize()
    payload = WorkflowResponse(workflow_id='frozen', finalized_demo=True).model_dump(mode='json')
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO analyses(id,user_id,repository,status,created_at,updated_at,result,events) VALUES('frozen','owner','repo','COMPLETE',1,1,:result,'[]')"), {'result': json.dumps(payload)})
    for field, value in [('repair', {}), ('finalized_demo', False), ('analysis', {'stale': True})]:
        stale = {**payload, field: value}
        with pytest.raises(IntegrityError, match='immutable'), engine.begin() as conn:
            conn.execute(text("UPDATE analyses SET result=:result WHERE id='frozen'"), {'result': json.dumps(stale)})
    with engine.begin() as conn:
        selected = {**payload, 'selection': {'variant': 'AI_CANDIDATE'}}
        conn.execute(text("UPDATE analyses SET result=:result WHERE id='frozen'"), {'result': json.dumps(selected)})
        assert json.loads(conn.execute(text("SELECT result FROM analyses WHERE id='frozen'")).scalar())['repair'] == payload['repair']
    engine.dispose()


def test_product_preserves_final_evidence_during_selection():
    saved = WorkflowResponse(workflow_id='frozen', finalized_demo=True).model_dump(mode='json')
    row = SimpleNamespace(result=copy.deepcopy(saved), events=[], updated_at=0)
    proposed = WorkflowResponse.model_validate(saved)
    proposed.selection = {'variant': 'ORIGINAL'}
    proposed.repair.explanation = 'stale replacement'
    product._persist_workflow(row, proposed, None, 'VARIANT_SELECTED')
    assert row.result['repair'] == saved['repair']
    assert row.result['selection']['variant'] == 'ORIGINAL'
    assert row.result['finalized_demo'] is True
    with pytest.raises(HTTPException, match='immutable'):
        product._persist_workflow(row, proposed, None, 'REPAIR_PROPOSAL_GENERATED')
