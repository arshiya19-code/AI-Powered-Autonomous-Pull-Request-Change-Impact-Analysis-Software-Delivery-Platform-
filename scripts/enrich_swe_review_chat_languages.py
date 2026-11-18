"""Enrich the cheap audit with cached GitHub primary-language metadata; never clones."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import requests

SUPPORTED = {"Python", "JavaScript", "TypeScript", "C#", "Rust", "C++"}

def main() -> None:
    root = Path("data/processed/swe_review_chat")
    repos = json.loads((root / "repository_summary.json").read_text(encoding="utf-8"))["per_repository"]
    cache_path = root / "github_repository_language_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set; run this command locally with a token.")
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    session = requests.Session()
    for repo in sorted(repos):
        if repo in cache: continue
        response = session.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=30)
        if response.status_code == 200:
            language = response.json().get("language")
            cache[repo] = {"primary_language": language, "mapped_language": language if language in SUPPORTED else None, "status": "ok"}
        else:
            cache[repo] = {"primary_language": None, "mapped_language": None, "status": f"http_{response.status_code}"}
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        time.sleep(0.05)
    (root / "repository_language_summary.json").write_text(json.dumps({"status": "completed", "requests": len(repos), "cached": len(cache), "supported": sorted(SUPPORTED), "repositories": cache}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "completed", "repositories": len(repos), "supported": sum(bool(v.get("mapped_language")) for v in cache.values()), "unsupported_or_unknown": sum(not v.get("mapped_language") for v in cache.values())}, indent=2))

if __name__ == "__main__": main()
