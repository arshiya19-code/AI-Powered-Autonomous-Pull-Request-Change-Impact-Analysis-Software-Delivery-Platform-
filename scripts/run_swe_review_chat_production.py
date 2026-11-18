"""Extract the training-ready SWE-Review-Chat production dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.swe_review_chat_production import DEFAULT_PRODUCTION_TARGET, run_production


def main() -> None:
    ap = argparse.ArgumentParser(description="SWE-Review-Chat production extraction")
    ap.add_argument("--target", type=int, default=DEFAULT_PRODUCTION_TARGET,
                    help="Accepted full-analysis rows for this milestone (default: 1000).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--git-timeout", type=int, default=120)
    ap.add_argument("--max-pr-runtime", type=float, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", type=Path, default=Path("data/processed/swe_review_chat"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.workers < 1 or args.git_timeout <= 0 or args.max_pr_runtime <= 0: raise SystemExit("workers, git-timeout, and max-pr-runtime must be positive")
    result = run_production(root=args.root, target=args.target, workers=args.workers, git_timeout=args.git_timeout,
                            max_pr_runtime=args.max_pr_runtime, seed=args.seed, resume=args.resume, token=os.environ.get("GITHUB_TOKEN"))
    print(result["report"])
    print("SWE-REVIEW-CHAT PRODUCTION DATASET PIPELINE COMPLETE")


if __name__ == "__main__": main()
