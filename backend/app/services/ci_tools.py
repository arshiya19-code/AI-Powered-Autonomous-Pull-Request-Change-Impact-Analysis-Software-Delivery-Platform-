"""Real CI scanners. Credentials stay in process environment, never evidence."""
from __future__ import annotations
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import httpx
from backend.app.schemas.models import ValidationCheck, normalize_validation_variant


def redact(value):
    text = str(value)
    for name, secret in os.environ.items():
        if secret and any(part in name.upper() for part in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY', 'PRIVATE_KEY')):
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----', '[REDACTED]', text, flags=re.S)
    text = re.sub(r'(?i)(bearer\s+|token\s+)[\w.\-+/=]+', r'\1[REDACTED]', text)
    return text


def executable(name):
    """Resolve a CLI like the user's shell, including Windows npm wrappers."""
    resolved = shutil.which(name)
    if resolved:
        return resolved
    appdata = os.environ.get('APPDATA', '')
    candidates = (
        Path(sys.executable).parent / (name + '.exe'),
        Path(sys.executable).parent / (name + '.cmd'),
        Path(sys.executable).parent / (name + '.ps1'),
        Path(appdata) / 'npm' / (name + '.cmd'),
        Path(appdata) / 'npm' / (name + '.ps1'),
    )
    for path in candidates:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            # An inaccessible PATH candidate is unavailable, not a readiness crash.
            continue
    return None


def _cli_command(path, *args):
    path = str(path)
    if path.lower().endswith('.ps1'):
        return ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path, *args]
    return [path, *args]


def command_env():
    # Repository test code does not need service credentials.
    return {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY', 'PRIVATE_KEY', 'GIT_CONFIG_'))}


class ScannerService:
    def __init__(self, client=None, run=None):
        self.client = client or httpx.Client(timeout=10, follow_redirects=False)
        self.run = run or subprocess.run

    def sonar_readiness(self):
        configured = bool(os.getenv('SONAR_HOST_URL') and os.getenv('SONAR_TOKEN') and executable('sonar-scanner'))
        if not configured:
            return {'status': 'NOT CONFIGURED', 'scanner_available': bool(executable('sonar-scanner')), 'credentials_configured': bool(os.getenv('SONAR_TOKEN'))}
        try:
            response = self.client.get(os.environ['SONAR_HOST_URL'].rstrip('/') + '/api/authentication/validate', auth=(os.environ['SONAR_TOKEN'], ''))
            response.raise_for_status()
            state = 'READY' if response.json().get('valid') else 'NOT CONFIGURED'
        except (httpx.HTTPError, ValueError):
            state = 'UNREACHABLE'
        return {'status': state, 'scanner_available': True, 'credentials_configured': True}

    def snyk_readiness(self):
        cli = executable('snyk')
        authenticated = bool(os.getenv('SNYK_TOKEN'))
        if cli and not authenticated:
            try:
                result = self.run(_cli_command(cli, 'config', 'get', 'api'), capture_output=True, text=True, timeout=10, shell=False, env=command_env())
                authenticated = result.returncode == 0 and result.stdout.strip() not in ('', 'undefined', 'null')
            except (OSError, subprocess.TimeoutExpired):
                pass
        return {'status': 'READY' if cli and authenticated else 'NOT CONFIGURED', 'scanner_available': bool(cli), 'authentication_configured': authenticated}

    def sonar(self, root, variant='AI_CANDIDATE', identity=None):
        variant = normalize_validation_variant(variant)
        readiness = self.sonar_readiness()
        if readiness['status'] != 'READY':
            configured = bool(os.getenv('SONAR_HOST_URL') and os.getenv('SONAR_TOKEN'))
            return ValidationCheck(check_type='sonarqube', status='UNAVAILABLE' if not configured else 'ERROR', tool='SonarQube', variant=variant, summary=readiness['status'], evidence={'state': readiness['status']})
        root = Path(root)
        identity = identity or {}
        # Unique project key prevents overlapping candidates from sharing metrics.
        import uuid
        key = 'impact-' + re.sub('[^A-Za-z0-9_-]', '-', str(identity.get('candidate_id') or uuid.uuid4()))
        command = [executable('sonar-scanner'), f'-Dsonar.projectKey={key}', '-Dsonar.sources=.', '-Dsonar.exclusions=**/.git/**,**/.venv/**,**/node_modules/**', '-Dsonar.qualitygate.wait=true']
        if (root / 'coverage.xml').is_file():
            command.append('-Dsonar.python.coverage.reportPaths=coverage.xml')
        started = time.monotonic()
        try:
            env = command_env()
            env.update(SONAR_TOKEN=os.environ['SONAR_TOKEN'], SONAR_HOST_URL=os.environ['SONAR_HOST_URL'])
            result = self.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=360, shell=False)
            report = root / '.scannerwork' / 'report-task.txt'
            data = dict(line.split('=', 1) for line in report.read_text().splitlines() if '=' in line)
            url = os.environ['SONAR_HOST_URL'].rstrip('/')
            auth = (os.environ['SONAR_TOKEN'], '')
            def get(path, params):
                response = self.client.get(url + path, params=params, auth=auth)
                response.raise_for_status()
                return response.json()
            task = get('/api/ce/task', {'id': data['ceTaskId']})['task']
            if task.get('status') != 'SUCCESS':
                raise ValueError('Sonar analysis did not finish')
            gate = get('/api/qualitygates/project_status', {'analysisId': task['analysisId']})['projectStatus']['status']
            metrics = get('/api/measures/component', {'component': key, 'metricKeys': 'bugs,vulnerabilities,security_hotspots,code_smells,coverage'})['component'].get('measures', [])
            measures = {m['metric']: float(m['value']) for m in metrics if 'value' in m}
            state = 'PASSED' if gate == 'OK' and result.returncode == 0 else 'FAILED'
            if gate not in ('OK', 'ERROR'):
                state = 'ERROR'
            return ValidationCheck(check_type='sonarqube', status=state, tool='SonarQube', variant=variant, workspace=str(root), duration_seconds=time.monotonic()-started, summary=f'SonarQube quality gate: {gate}', evidence={'quality_gate': gate, 'measures': measures, 'analysis_id': task['analysisId']})
        except (OSError, ValueError, KeyError, httpx.HTTPError, subprocess.TimeoutExpired):
            return ValidationCheck(check_type='sonarqube', status='ERROR', tool='SonarQube', variant=variant, summary='Configured SonarQube scan did not produce complete quality-gate evidence.')

    def _snyk_report(self, root):
        env = command_env()
        if os.getenv('SNYK_TOKEN'):
            env['SNYK_TOKEN'] = os.environ['SNYK_TOKEN']
        result = self.run(_cli_command(executable('snyk'), 'test', '--all-projects', '--json', '--ignore-policy'), cwd=root, env=env, capture_output=True, text=True, timeout=300, shell=False)
        if result.returncode not in (0, 1):
            raise ValueError('Snyk scan failed')
        reports = json.loads(result.stdout)
        reports = reports if isinstance(reports, list) else [reports]
        if not reports or any(not isinstance(r, dict) or r.get('error') or 'vulnerabilities' not in r for r in reports):
            raise ValueError('Incomplete Snyk report')
        findings = {}
        for report in reports:
            target = str(report.get('targetFile', '')).replace('\\', '/')
            if Path(target).is_absolute():
                target = Path(target).resolve().relative_to(Path(root).resolve()).as_posix()
            for item in report['vulnerabilities']:
                severity = item['severity'].lower()
                if severity not in ('critical', 'high', 'medium', 'low'):
                    raise ValueError('Unknown severity')
                dependency_path = item.get('from', [])[1:]  # root project version is not a dependency
                key = json.dumps([target, item['id'], item.get('packageName'), item.get('version'), dependency_path])
                findings[key] = severity
        return findings

    @staticmethod
    def _manifest_snapshot(root):
        root = Path(root).resolve()
        names = {
            'requirements.txt', 'requirements-dev.txt', 'pyproject.toml', 'poetry.lock',
            'Pipfile', 'Pipfile.lock', 'package.json', 'package-lock.json', 'yarn.lock',
            'pnpm-lock.yaml', 'Cargo.toml', 'Cargo.lock',
        }
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*')
            if path.is_file() and path.name in names and '.git' not in path.parts
        }

    @staticmethod
    def _valid_host_snyk_evidence(path):
        if not path:
            return False
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError):
            return False
        reports = payload if isinstance(payload, list) else [payload]
        return bool(reports) and all(
            isinstance(report, dict)
            and not report.get('error')
            and isinstance(report.get('vulnerabilities'), list)
            for report in reports
        )

    def snyk(self, root, baseline_root=None, variant='AI_CANDIDATE'):
        variant = normalize_validation_variant(variant)
        ready = self.snyk_readiness()
        if ready['status'] != 'READY':
            return ValidationCheck(check_type='snyk', status='UNAVAILABLE', tool='Snyk', variant=variant, summary='NOT CONFIGURED', evidence={'state': 'NOT CONFIGURED'})
        started = time.monotonic()
        try:
            current = self._snyk_report(root)
            baseline = self._snyk_report(baseline_root) if baseline_root else None
            if variant == 'AI_CANDIDATE' and baseline is None:
                raise ValueError('Candidate baseline required')
            introduced = {key: level for key, level in current.items() if baseline is not None and (key not in baseline or (level in ('critical', 'high') and baseline[key] not in ('critical', 'high')))}
            counts = {level: sum(s == level for s in current.values()) for level in ('critical', 'high', 'medium', 'low')}
            added = {level: sum(s == level for s in introduced.values()) for level in counts}
            failed = added['critical'] + added['high'] > 0
            return ValidationCheck(check_type='snyk', status='FAILED' if failed else 'PASSED', tool='Snyk', variant=variant, workspace=str(root), severity_counts=counts, duration_seconds=time.monotonic()-started, summary='Snyk dependency scan: introduced Critical/High block; Medium/Low advisory.', evidence={'introduced': added, 'baseline_compared': baseline is not None})
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
            return ValidationCheck(check_type='snyk', status='ERROR', tool='Snyk', variant=variant, summary='Configured Snyk scan or baseline comparison failed; no clean result claimed.')


def tool_readiness():
    scanner = ScannerService()
    # Only active prototype checks are exposed to the product UI.  The scanner
    # service still owns Snyk readiness and its real baseline comparison.
    return {
        'Python Compilation': {'status': 'READY'},
        'PyTest': {'status': 'READY' if importlib.util.find_spec('pytest') else 'NOT CONFIGURED'},
        'Ruff': {'status': 'READY' if executable('ruff') else 'NOT CONFIGURED'},
        'Bandit': {'status': 'READY' if importlib.util.find_spec('bandit') else 'NOT CONFIGURED'},
        'Snyk': scanner.snyk_readiness(),
        'Dependency Manifest': {'status': 'READY'},
        'Candidate Re-analysis': {'status': 'READY'},
    }
