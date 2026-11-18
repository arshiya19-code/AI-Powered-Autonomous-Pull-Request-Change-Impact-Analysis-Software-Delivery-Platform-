"""Jenkins executes validation; the backend owns identity and all business gates."""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from uuid import uuid4
import httpx
from backend.app.schemas.models import ValidationCheck, normalize_validation_variant
from .ci_tools import redact

STAGES = ('build', 'unit', 'regression', 'targeted_integration', 'security', 'sonarqube', 'snyk', 'dependency')


class JenkinsService:
    def __init__(self, client=None):
        self.url = os.getenv('JENKINS_URL', '').rstrip('/')
        self.job = os.getenv('JENKINS_JOB', '')
        self.user = os.getenv('JENKINS_USER', '')
        self.token = os.getenv('JENKINS_TOKEN', '')
        self.shared = os.getenv('CI_SHARED_ROOT', '')
        self.client = client or httpx.Client(timeout=15, follow_redirects=False)

    @property
    def job_url(self):
        return self.url + '/' + '/'.join('job/' + quote(part, safe='') for part in self.job.split('/'))

    def _request(self, method, url, **kwargs):
        # Never send credentials to a different host supplied by queue JSON.
        if not url.startswith(self.url + '/') or urlparse(url).netloc != urlparse(self.url).netloc:
            raise ValueError('Jenkins returned an unexpected URL')
        response = self.client.request(method, url, auth=(self.user, self.token), **kwargs)
        response.raise_for_status()
        return response

    def readiness(self):
        safe = {'url': self.url if self.url and not (urlparse(self.url).username or urlparse(self.url).query or urlparse(self.url).fragment) else None, 'credentials_configured': bool(self.user and self.token), 'api_usable': False}
        if not all((self.url, self.job, self.user, self.token, self.shared)):
            return {**safe, 'status': 'NOT CONFIGURED'}
        try:
            job = self._request('GET', self.job_url + '/api/json', params={'tree': 'name,buildable'}).json()
            if not job.get('buildable', False):
                return {**safe, 'status': 'NOT CONFIGURED', 'api_usable': True}
            return {**safe, 'status': 'READY', 'api_usable': True}
        except (httpx.HTTPError, ValueError):
            return {**safe, 'status': 'UNREACHABLE'}

    @staticmethod
    def ingest(payload, identity):
        from .validation import ValidationService
        if any(payload.get(k) != identity.get(k) for k in ('analysis_id', 'candidate_id', 'attempt', 'bundle_sha256', 'base_sha', 'head_sha')):
            raise ValueError('Jenkins evidence identity mismatch')
        if set(payload.get('checks', {})) != set(STAGES) or payload.get('handoff') != 'BACKEND_REANALYSIS_REQUIRED':
            raise ValueError('Jenkins evidence is incomplete')
        checks = {key: ValidationCheck.model_validate(value) for key, value in payload['checks'].items()}
        for key, check in checks.items():
            if any(not isinstance(n, int) or n < 0 for n in check.severity_counts.values()):
                raise ValueError('Invalid severity evidence')
            if check.check_type != key or check.variant != 'AI_CANDIDATE':
                raise ValueError('Jenkins check identity mismatch')
            if key == 'sonarqube' and check.status == 'PASSED' and check.evidence.get('quality_gate') != 'OK':
                raise ValueError('SonarQube quality gate evidence missing')
            if key == 'snyk' and check.status == 'PASSED':
                evidence = check.evidence
                if not evidence.get('baseline_compared') or not all(k in evidence.get('introduced', {}) for k in ('critical', 'high', 'medium', 'low')):
                    raise ValueError('Snyk comparison evidence missing')
                if any(not isinstance(n, int) or n < 0 for n in evidence['introduced'].values()):
                    raise ValueError('Invalid Snyk severity evidence')
                if evidence['introduced']['critical'] or evidence['introduced']['high']:
                    check.status = 'FAILED'
            if key in ('build', 'unit', 'regression') and check.status not in ('PASSED', 'FAILED', 'ERROR'):
                check.status = 'ERROR'
        summary = ValidationService().aggregate(checks)
        summary.ci = {**identity, 'executor': 'jenkins', 'status': 'PASS' if summary.status == 'passed' else 'FAIL'}
        return summary

    def validate(self, root, baseline_root, identity, regression, integration):
        from .validation import ValidationService
        state = self.readiness()['status']
        correlation = {**identity, 'executor': 'jenkins', 'status': state}
        def failed(message, status='FAIL'):
            summary = ValidationService().aggregate({'jenkins': ValidationCheck(check_type='jenkins', status='ERROR', variant='AI_CANDIDATE', summary=message)})
            summary.ci = {**correlation, 'status': status}
            return summary
        if state != 'READY':
            return failed('Jenkins is ' + state + '. Select CI_EXECUTOR=local explicitly to use local validation.', state)
        try:
            bundle_id = str(uuid4())
            bundle = Path(self.shared).resolve() / bundle_id
            bundle.mkdir(parents=True)
            ignore = shutil.ignore_patterns('.git', '.env', '.env.*', '*.pem', '*.key', '*secret*', '.venv', 'node_modules', '__pycache__', '.scannerwork')
            shutil.copytree(root, bundle / 'candidate', ignore=ignore, symlinks=True)
            shutil.copytree(baseline_root, bundle / 'baseline', ignore=ignore, symlinks=True)
            digest = bundle_digest(bundle)
            correlation['bundle_sha256'] = digest
            manifest = {**correlation, 'regression': regression, 'targeted_integration': integration}
            (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            response = self._request('POST', self.job_url + '/buildWithParameters', data={'BUNDLE_ID': bundle_id, 'BUNDLE_SHA256': digest})
            queue = response.headers.get('Location')
            if not queue:
                raise ValueError('Jenkins did not return a queue URL')
            deadline = time.monotonic() + int(os.getenv('JENKINS_TIMEOUT', '1200'))
            build_url = None
            correlation['status'] = 'RUNNING'
            while time.monotonic() < deadline:
                if build_url is None:
                    item = self._request('GET', queue.rstrip('/') + '/api/json').json()
                    if item.get('cancelled'):
                        return failed('Jenkins queue item was cancelled.')
                    build_url = item.get('executable', {}).get('url')
                else:
                    build = self._request('GET', build_url.rstrip('/') + '/api/json').json()
                    if not build.get('building', True):
                        if build.get('result') != 'SUCCESS':
                            return failed('Jenkins execution failed; candidate validation is blocked.')
                        payload = self._request('GET', build_url.rstrip('/') + '/artifact/ci-result.json').json()
                        # Defense in depth: remove any configured secret before persistence.
                        payload = json.loads(redact(json.dumps(payload)))
                        summary = self.ingest(payload, correlation)
                        summary.ci['build_url'] = build_url
                        return summary
                time.sleep(2)
            return failed('Jenkins wait timed out. The external build may still be running; candidate remains blocked.', 'RUNNING')
        except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
            return failed('Jenkins submission, polling, or correlated evidence ingestion failed.')


def bundle_digest(bundle):
    digest = hashlib.sha256()
    for folder in ('candidate', 'baseline'):
        for path in sorted((Path(bundle) / folder).rglob('*')):
            if path.is_symlink():
                raise ValueError('Candidate bundle contains a symbolic link')
            if path.is_file():
                digest.update(path.relative_to(bundle).as_posix().encode() + b'\0')
                digest.update(path.read_bytes())
    return digest.hexdigest()
