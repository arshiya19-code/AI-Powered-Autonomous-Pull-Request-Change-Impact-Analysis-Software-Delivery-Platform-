"""Administrator-only platform views over persisted account and analysis data."""
import time
from datetime import datetime, timezone, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select, text

from . import auth
from .auth import admin
from .database import Analysis, AuditEvent, Session, User, public_user
from .presentation import build_presentation

router = APIRouter(prefix='/api/admin', dependencies=[Depends(admin)])

def _analysis_values(row: Analysis):
    workflow = row.result or {}
    analysis = workflow.get('analysis') or workflow
    risk = analysis.get('risk_prediction') or {}
    validation = workflow.get('validation') or analysis.get('validation') or {}
    decision = workflow.get('decision') or analysis.get('decision') or {}
    deployment = workflow.get('deployment') or analysis.get('deployment') or {}
    events = row.events or []
    probability = risk.get('risk_probability')
    risk_class = risk.get('risk_class')
    if risk_class is not None:
        predicted = 'HIGH' if risk_class in (1, '1', True, 'HIGH') else 'LOW'
    elif probability is not None:
        predicted = 'HIGH' if probability >= (risk.get('threshold') if risk.get('threshold') is not None else .5) else 'LOW'
    else:
        predicted = None
    return {
        'id': row.id,
        'user_id': row.user_id,
        'repository': row.repository,
        'pull_request': row.pull_request,
        'created_at': row.created_at,
        'updated_at': row.updated_at or row.created_at,
        'workflow_state': events[-1].get('type') if events else row.status,
        'status': row.status,
        'risk_probability': probability,
        'predicted_risk_class': predicted,
        'risk_model_status': risk.get('model_status'),
        'validation_status': validation.get('status'),
        'recommendation': decision.get('decision') or deployment.get('recommended_action'),
        'approval_state': workflow.get('approval_state'),
        'deployment_state': deployment.get('lifecycle_state') or deployment.get('simulation_status') or deployment.get('status'),
    }

def _activity_values(event: AuditEvent, users: dict[str, User]):
    actor = users.get(event.actor_id)
    subject = users.get(event.subject_id or '')
    return {
        'id': event.id,
        'actor_id': event.actor_id,
        'actor_name': actor.display_name if actor else None,
        'actor_email': actor.email if actor else None,
        'subject_id': event.subject_id,
        'subject_name': subject.display_name if subject else None,
        'action': event.action,
        'created_at': event.created_at,
        'metadata': event.details or {},
    }

def _user_rollups(db):
    counts = dict(db.execute(select(Analysis.user_id, func.count()).group_by(Analysis.user_id)).all())
    latest_analyses = dict(db.execute(select(Analysis.user_id, func.max(func.coalesce(Analysis.updated_at, Analysis.created_at))).group_by(Analysis.user_id)).all())
    latest_audits = dict(db.execute(select(AuditEvent.actor_id, func.max(AuditEvent.created_at)).group_by(AuditEvent.actor_id)).all())
    return counts, latest_analyses, latest_audits

def _user_values(user: User, counts, latest_analyses, latest_audits):
    activity = max(filter(lambda value: value is not None, [user.last_login, user.updated_at, latest_analyses.get(user.id), latest_audits.get(user.id)]), default=user.created_at)
    return {**public_user(user), 'analysis_count': counts.get(user.id, 0), 'latest_activity_at': activity}

@router.get('/overview')
def overview():
    now = time.time()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    week = (datetime.now(timezone.utc) - timedelta(days=7)).timestamp()
    with auth.DB() as db:
        users = list(db.scalars(select(User)))
        analyses = list(db.scalars(select(Analysis)))
        user_map = {u.id: u for u in users}
        counts, latest_analyses, latest_audits = _user_rollups(db)
        stats = {
            'total_users': len(users),
            'active_users': sum(u.is_active for u in users),
            'inactive_users': sum(not u.is_active for u in users),
            'admins': sum(u.role == 'ADMIN' for u in users),
            'normal_users': sum(u.role == 'USER' for u in users),
            'total_analyses': len(analyses),
            'analyses_today': sum(a.created_at >= today for a in analyses),
            'analyses_this_week': sum(a.created_at >= week for a in analyses),
            'completed_analyses': sum(a.status == 'COMPLETE' for a in analyses),
            'failed_analyses': sum(a.status == 'FAILED' for a in analyses),
            'awaiting_approval': sum(_analysis_values(a)['approval_state'] == 'READY_FOR_APPROVAL' for a in analyses),
        }
        recent_users = [_user_values(u, counts, latest_analyses, latest_audits) for u in sorted(users, key=lambda item: item.created_at, reverse=True)[:8]]
        recent_analyses = []
        for row in sorted(analyses, key=lambda item: item.updated_at or item.created_at, reverse=True)[:10]:
            values = _analysis_values(row); owner = user_map.get(row.user_id)
            recent_analyses.append({**values, 'user': public_user(owner) if owner else None})
        return {'stats': stats, 'recent_registrations': recent_users, 'recent_analyses': recent_analyses, 'system_health': {'database': 'connected', 'accounts': 'available', 'analyses': 'available', 'checked_at': now}}

@router.get('/users')
def users():
    with auth.DB() as db:
        counts, latest_analyses, latest_audits = _user_rollups(db)
        return {'items': [_user_values(u, counts, latest_analyses, latest_audits) for u in db.scalars(select(User).order_by(User.created_at.desc()).limit(500))]}

@router.get('/users/{user_id}')
def user_detail(user_id: str):
    with auth.DB() as db:
        account = db.get(User, user_id)
        if not account: raise HTTPException(404, 'User not found')
        counts, latest_analyses, latest_audits = _user_rollups(db)
        analyses = list(db.scalars(select(Analysis).where(Analysis.user_id == user_id).order_by(Analysis.created_at.desc()).limit(25)))
        events = list(db.scalars(select(AuditEvent).where(or_(AuditEvent.actor_id == user_id, AuditEvent.subject_id == user_id)).order_by(AuditEvent.created_at.desc()).limit(100)))
        all_users = {u.id: u for u in db.scalars(select(User))}
        return {
            'profile': _user_values(account, counts, latest_analyses, latest_audits),
            'usage': {'analysis_count': counts.get(user_id, 0), 'recent_analyses': [_analysis_values(row) for row in analyses]},
            'activity': [_activity_values(event, all_users) for event in events],
        }

class AdminUpdate(BaseModel):
    role: Literal['USER', 'ADMIN']
    is_active: bool

@router.patch('/users/{user_id}')
def update_user(user_id: str, body: AdminUpdate, actor=Depends(admin)):
    if user_id == actor.id: raise HTTPException(409, 'You cannot change your own administrator access')
    with auth.DB() as db:
        row = db.get(User, user_id)
        if not row: raise HTTPException(404, 'User not found')
        previous = {'role': row.role, 'is_active': row.is_active}
        row.role = body.role; row.is_active = body.is_active; row.updated_at = time.time()
        db.execute(delete(Session).where(Session.user_id == user_id))
        actions = []
        if previous['role'] != body.role: actions.append('ROLE_CHANGED')
        if previous['is_active'] != body.is_active: actions.append('ACCOUNT_ACTIVATED' if body.is_active else 'ACCOUNT_DEACTIVATED')
        if not actions: actions.append('ACCOUNT_ACCESS_UPDATED')
        for action in actions:
            db.add(AuditEvent(actor_id=actor.id, action=action, subject_id=user_id, details={'role': body.role, 'is_active': body.is_active}))
        db.commit()
        return public_user(row)

@router.get('/analyses')
def analyses(user_id: str | None = None, status: str | None = None, date: str | None = None, repository: str | None = None, risk_state: str | None = None, limit: int = Query(200, ge=1, le=500)):
    with auth.DB() as db:
        query = select(Analysis)
        if user_id: query = query.where(Analysis.user_id == user_id)
        if status: query = query.where(Analysis.status == status.upper())
        if repository: query = query.where(func.lower(Analysis.repository).contains(repository.strip().lower()))
        if date:
            try:
                start = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
            except ValueError: raise HTTPException(422, 'Date must use YYYY-MM-DD')
            query = query.where(Analysis.created_at >= start, Analysis.created_at < start + 86400)
        rows = list(db.scalars(query.order_by(Analysis.created_at.desc()).limit(limit)))
        users = {u.id: u for u in db.scalars(select(User))}
        items = []
        for row in rows:
            values = _analysis_values(row)
            if risk_state and risk_state.upper() not in {str(values['predicted_risk_class']).upper(), str(values['risk_model_status']).upper()}:
                continue
            owner = users.get(row.user_id)
            items.append({**values, 'user': public_user(owner) if owner else None})
        return {'items': items, 'total': len(items)}

@router.get('/analyses/{analysis_id}')
def analysis_detail(analysis_id: str):
    with auth.DB() as db:
        row = db.get(Analysis, analysis_id)
        if not row: raise HTTPException(404, 'Analysis not found')
        owner = db.get(User, row.user_id)
        return {**_analysis_values(row), 'user': public_user(owner) if owner else None, 'events': row.events or [], 'result': row.result, 'presentation': build_presentation(row.id, row.status, row.result or {})}

@router.get('/activity')
def activity(limit: int = Query(200, ge=1, le=500), action: str | None = None):
    with auth.DB() as db:
        query = select(AuditEvent)
        if action: query = query.where(AuditEvent.action == action.upper())
        events = list(db.scalars(query.order_by(AuditEvent.created_at.desc()).limit(limit)))
        users = {u.id: u for u in db.scalars(select(User))}
        return {'items': [_activity_values(event, users) for event in events]}
