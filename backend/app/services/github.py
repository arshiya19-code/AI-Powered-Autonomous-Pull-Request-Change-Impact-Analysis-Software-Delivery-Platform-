"""GitHub App boundary. Installation tokens are short lived and never persisted."""
from __future__ import annotations

import os
import re
import subprocess
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

SAFE_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SOURCE_SUFFIXES = {'.py','.pyi','.js','.jsx','.ts','.tsx','.java','.kt','.kts','.cs','.go','.rs','.c','.h','.cc','.cpp','.hpp','.rb','.php','.swift','.scala'}

class GitHubIntegrationError(RuntimeError):
    def __init__(self, message: str, code: str = 'github_error', status_code: int = 502):
        super().__init__(message); self.code = code; self.status_code = status_code

def _timestamp(value: str | None) -> float | None:
    if not value: return None
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()

class GitHubService:
    api = 'https://api.github.com'
    def __init__(self, app_id: str | None = None, private_key: str | None = None, client: httpx.Client | None = None):
        self.app_id = app_id or os.getenv('GITHUB_APP_ID')
        configured_key = private_key or os.getenv('GITHUB_APP_PRIVATE_KEY')
        key_path = os.getenv('GITHUB_APP_PRIVATE_KEY_PATH')
        self.private_key = configured_key.replace('\\n','\n') if configured_key else (Path(key_path).read_text() if key_path and Path(key_path).is_file() else None)
        self.slug = os.getenv('GITHUB_APP_SLUG')
        self.client = client or httpx.Client(timeout=30, follow_redirects=False)
        self._tokens: dict[str, tuple[str,float]] = {}

    @property
    def configured(self) -> bool: return bool(self.app_id and self.private_key and self.slug)

    def installation_url(self, state: str) -> str:
        if not self.slug: raise GitHubIntegrationError('GitHub App is not configured.', 'not_configured', 503)
        from urllib.parse import urlencode
        return f'https://github.com/apps/{self.slug}/installations/new?{urlencode({"state": state})}'

    def _app_jwt(self) -> str:
        if not self.app_id or not self.private_key: raise GitHubIntegrationError('GitHub App credentials are not configured.', 'not_configured', 503)
        try:
            import jwt
            now = int(time.time())
            return jwt.encode({'iat': now-30, 'exp': now+540, 'iss': self.app_id}, self.private_key, algorithm='RS256')
        except Exception as exc: raise GitHubIntegrationError('GitHub App authentication failed.', 'authentication_failed') from exc

    def installation_token(self, installation_id: str) -> str:
        cached = self._tokens.get(str(installation_id))
        if cached and cached[1] > time.time()+60: return cached[0]
        response = self.client.post(f'{self.api}/app/installations/{installation_id}/access_tokens', headers={'Authorization':f'Bearer {self._app_jwt()}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
        data = self._response(response)
        token = data.get('token')
        if not token: raise GitHubIntegrationError('GitHub did not return an installation token.', 'authentication_failed')
        expires = _timestamp(data.get('expires_at')) or time.time()+3300
        self._tokens[str(installation_id)] = (token, expires)
        return token

    def _response(self, response: httpx.Response) -> Any:
        if response.status_code == 403 and response.headers.get('x-ratelimit-remaining') == '0':
            raise GitHubIntegrationError('GitHub API rate limit reached. Try again after the reset time.', 'rate_limited', 429)
        if response.status_code in (401,403,404):
            raise GitHubIntegrationError('GitHub repository access is unavailable or has been revoked.', 'access_revoked', 403)
        if response.is_error: raise GitHubIntegrationError(f'GitHub API request failed ({response.status_code}).')
        return response.json()

    def _get(self, installation_id: str, path: str, params: dict | None = None) -> Any:
        return self._response(self.client.get(f'{self.api}{path}', params=params, headers={'Authorization':f'Bearer {self.installation_token(installation_id)}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}))

    def list_authorized_repositories(self, installation_id: str) -> list[dict]:
        repositories=[]; page=1
        while True:
            data=self._get(installation_id, '/installation/repositories', {'per_page':100,'page':page})
            batch=data.get('repositories',[]); repositories.extend(batch)
            if len(batch)<100: break
            page += 1
        return repositories

    def list_open_pull_requests(self, installation_id: str, full_name: str) -> list[dict]:
        if not SAFE_REPOSITORY.fullmatch(full_name): raise GitHubIntegrationError('Repository identity is invalid.', 'invalid_repository', 422)
        items=[]; page=1
        while True:
            batch=self._get(installation_id, f'/repos/{full_name}/pulls', {'state':'open','per_page':100,'page':page}); items.extend(batch)
            if len(batch)<100: return items
            page += 1

    def get_pull_request(self, installation_id: str, full_name: str, number: int) -> dict:
        if not SAFE_REPOSITORY.fullmatch(full_name): raise GitHubIntegrationError('Repository identity is invalid.', 'invalid_repository', 422)
        return self._get(installation_id, f'/repos/{full_name}/pulls/{number}')

    def get_changed_files(self, installation_id: str, full_name: str, number: int) -> list[dict]:
        items=[]; page=1
        while True:
            batch=self._get(installation_id, f'/repos/{full_name}/pulls/{number}/files', {'per_page':100,'page':page}); items.extend(batch)
            if len(batch)<100: return items
            page += 1

    @staticmethod
    def repository_values(item: dict) -> dict:
        owner=(item.get('owner') or {}).get('login') or item.get('full_name','').partition('/')[0]
        return {'repository_id':str(item['id']), 'owner':owner, 'repo_name':item['name'], 'full_name':item['full_name'], 'repository':item['full_name'], 'default_branch':item.get('default_branch') or 'main', 'connection_status':'CONNECTED', 'last_synced_at':time.time()}

    @staticmethod
    def pull_request_values(item: dict) -> dict:
        merged = bool(item.get('merged') or item.get('merged_at'))
        return {'provider_pr_id':str(item['id']), 'pr_number':int(item['number']), 'title':item.get('title') or f"PR #{item['number']}", 'state':'merged' if merged else item.get('state','open'), 'author':(item.get('user') or {}).get('login'), 'base_sha':(item.get('base') or {}).get('sha') or '', 'head_sha':(item.get('head') or {}).get('sha') or '', 'base_branch':(item.get('base') or {}).get('ref'), 'head_branch':(item.get('head') or {}).get('ref'), 'html_url':item.get('html_url'), 'opened_at':_timestamp(item.get('created_at')), 'updated_at':_timestamp(item.get('updated_at')), 'merged_at':_timestamp(item.get('merged_at')), 'last_synced_at':time.time()}

    def materialize_pull_request(self, installation_id: str, full_name: str, base_sha: str, head_sha: str, workspace: str | Path) -> tuple[str,dict[str,str],list[dict]]:
        if not SAFE_REPOSITORY.fullmatch(full_name) or not re.fullmatch(r'[0-9a-fA-F]{7,64}',base_sha or '') or not re.fullmatch(r'[0-9a-fA-F]{7,64}',head_sha or ''):
            raise GitHubIntegrationError('Repository or commit identity is invalid.', 'invalid_repository', 422)
        root=Path(workspace).resolve(); root.mkdir(parents=True,exist_ok=True)
        token=self.installation_token(installation_id)
        # GitHub's REST API accepts bearer installation tokens, but Git's smart
        # HTTP transport authenticates installation tokens as x-access-token
        # Basic credentials. Keep the credential in the child environment so it
        # is never written to the remote URL, command line, or persisted result.
        credential=base64.b64encode(f'x-access-token:{token}'.encode()).decode()
        env=os.environ.copy(); env.update({'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'http.https://github.com/.extraheader','GIT_CONFIG_VALUE_0':f'Authorization: Basic {credential}','GIT_TERMINAL_PROMPT':'0'})
        stage='initialize'
        def git(*args: str) -> str:
            try:
                result=subprocess.run(['git',*args],cwd=root,env=env,capture_output=True,text=True,timeout=120,check=False)
            except (OSError,subprocess.TimeoutExpired) as exc: raise GitHubIntegrationError('Repository checkout failed.', 'checkout_failed') from exc
            if result.returncode:
                detail=(result.stderr or result.stdout or '').strip().splitlines()
                reason=detail[-1][:300] if detail else 'git command returned a non-zero exit code'
                raise GitHubIntegrationError(f'Repository checkout failed during {stage}: {reason}', 'checkout_failed')
            return result.stdout
        git('init','--quiet'); git('remote','add','origin',f'https://github.com/{full_name}.git')
        stage='base fetch'; git('fetch','--quiet','--no-tags','--depth=1','origin',base_sha)
        stage='head fetch'; git('fetch','--quiet','--no-tags','--depth=1','origin',head_sha)
        stage='head checkout'; git('checkout','--quiet','--detach',head_sha)
        stage='diff';
        diff=git('diff','--no-ext-diff','--binary',base_sha,head_sha)
        # An empty base/head diff is valid evidence. The analysis layer records it
        # as EMPTY_CHANGE/NO_CHANGES; it must not be confused with checkout or
        # graph failure.
        sources={}; total=0
        for path in root.rglob('*'):
            if not path.is_file() or '.git' in path.parts or path.suffix.lower() not in SOURCE_SUFFIXES: continue
            try:
                size=path.stat().st_size
                if size>500_000 or total+size>5_000_000 or len(sources)>=1000: continue
                raw=path.read_bytes()
                if b'\x00' in raw: continue
                sources[path.relative_to(root).as_posix()]=raw.decode('utf-8',errors='replace'); total += size
            except OSError: continue
        if not sources and diff.strip(): raise GitHubIntegrationError('The pull request contains only unsupported or binary changes.', 'unsupported_changes', 422)
        changed_files=[line.split(' b/',1)[1] for line in diff.splitlines() if line.startswith('diff --git a/') and ' b/' in line]
        evidence=[
            {'type':'GITHUB_INSTALLATION_TOKEN_SUCCEEDED','installation_id':str(installation_id)},
            {'type':'REPOSITORY_CHECKOUT_SUCCEEDED','base_sha':base_sha,'head_sha':head_sha},
            {'type':'PR_DIFF_ACQUIRED','changed_files':changed_files,'changed_file_count':len(changed_files),'diff_bytes':len(diff.encode())},
            {'type':'SOURCE_SNAPSHOT_ACQUIRED','source_file_count':len(sources)},
        ]
        return diff,sources,evidence
