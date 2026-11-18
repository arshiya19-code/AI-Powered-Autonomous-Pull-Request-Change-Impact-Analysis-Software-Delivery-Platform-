"""Durable stage transitions shared by background jobs and presentation."""
import time

NAMES = ('PR_ACQUIRED', 'CODE_PARSED', 'GRAPH_BUILT', 'IMPACT', 'RISK', 'EXPLANATION',
         'DECISION', 'VALIDATION', 'REPAIR', 'TEST_GENERATION', 'CANDIDATE_COMPARISON',
         'USER_APPROVAL', 'DEPLOYMENT', 'MONITORING', 'FEEDBACK')
TRANSITIONS = {
    'PR_ACQUISITION_STARTED': ('PR_ACQUIRED', 'RUNNING'),
    'PR_DIFF_ACQUIRED': ('PR_ACQUIRED', 'PASS'),
    'ANALYSIS_STARTED': ('CODE_PARSED', 'RUNNING'),
    'SOURCE_PARSED': ('CODE_PARSED', 'PASS'),
    'GRAPH_STARTED': ('GRAPH_BUILT', 'RUNNING'),
    'GRAPH_BUILT': ('GRAPH_BUILT', 'PASS'),
    'GRAPH_FAILED': ('GRAPH_BUILT', 'FAILED'),
    'GRAPH_UNAVAILABLE': ('GRAPH_BUILT', 'NOT_DEFINED'),
    'IMPACT_STARTED': ('IMPACT', 'RUNNING'),
    'IMPACT_ANALYZED': ('IMPACT', 'PASS'),
    'RISK_STARTED': ('RISK', 'RUNNING'),
    'RISK_COMPLETE': ('RISK', 'PASS'),
    'RISK_UNAVAILABLE': ('RISK', 'NOT_CONFIGURED'),
    'SHAP_STARTED': ('EXPLANATION', 'RUNNING'),
    'SHAP_COMPLETE': ('EXPLANATION', 'PASS'),
    'SHAP_UNAVAILABLE': ('EXPLANATION', 'NOT_CONFIGURED'),
    'DECISION_STARTED': ('DECISION', 'RUNNING'),
    'DECISION_COMPLETE': ('DECISION', 'PASS'),
    'VALIDATION_STARTED': ('VALIDATION', 'RUNNING'),
    'VALIDATION_COMPLETE': ('VALIDATION', 'PASS'),
    'VALIDATION_UNAVAILABLE': ('VALIDATION', 'NOT_DEFINED'),
}

def transition(stages, kind, details=None):
    now = time.time()
    stages = {name: dict((stages or {}).get(name) or {
        'stage': name, 'status': 'PENDING', 'started_at': None, 'completed_at': None,
        'duration': None, 'message': 'Awaiting execution', 'evidence': {},
    }) for name in NAMES}
    if kind in TRANSITIONS:
        name, status = TRANSITIONS[kind]
        if kind == 'VALIDATION_COMPLETE' and (details or {}).get('status') == 'failed': status = 'FAILED'
        item = stages[name]
        item.update(status=status, started_at=item.get('started_at') or now,
                    message=(details or {}).get('message') or kind.replace('_', ' ').capitalize(),
                    evidence=details or {})
        if status != 'RUNNING':
            item.update(completed_at=now, duration=now-item['started_at'])
        if status == 'FAILED':
            downstream = False
            for downstream_name in NAMES:
                if downstream_name == name:
                    downstream = True
                    continue
                if downstream:
                    downstream_item = stages[downstream_name]
                    if downstream_item['status'] in ('PENDING', 'RUNNING'):
                        downstream_item.update(status='SKIPPED', completed_at=now, duration=None,
                                                message=f'Not executed because {name} failed.')
    if kind == 'WORKFLOW_FAILED':
        active = next((name for name in NAMES if stages[name]['status'] == 'RUNNING'), None)
        active = active or next((name for name in NAMES if stages[name]['status'] == 'PENDING'), 'PR_ACQUIRED')
        for name, item in stages.items():
            if name == active or item['status'] in ('PENDING', 'RUNNING'):
                item.update(status='FAILED' if name == active else 'SKIPPED', completed_at=now,
                            duration=now-item['started_at'] if item.get('started_at') else None,
                            message=(details or {}).get('message') if name == active else f'Not executed because {active} failed.',
                            error=details if name == active else None)
    return stages
