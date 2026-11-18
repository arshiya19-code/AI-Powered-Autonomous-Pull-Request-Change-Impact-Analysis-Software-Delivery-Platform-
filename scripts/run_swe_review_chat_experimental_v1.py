"""Build only the 300-row, full-analysis Experimental Dataset V1."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.swe_review_chat_experimental import run_experimental_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-Review-Chat Experimental Dataset V1")
    parser.add_argument("--root", type=Path, default=Path("data/processed/swe_review_chat"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--git-timeout", type=int, default=120)
    parser.add_argument("--max-pr-runtime", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-expensive-analyses", type=int, default=None,
                        help="Bound a live smoke run; does not produce final 300-row artifacts.")
    args = parser.parse_args()
    print(run_experimental_v1(root=args.root, workers=args.workers, git_timeout=args.git_timeout,
                              max_pr_runtime=args.max_pr_runtime, seed=args.seed, token=os.environ.get("GITHUB_TOKEN"),
                              max_expensive_analyses=args.max_expensive_analyses))


if __name__ == "__main__":
    main()
