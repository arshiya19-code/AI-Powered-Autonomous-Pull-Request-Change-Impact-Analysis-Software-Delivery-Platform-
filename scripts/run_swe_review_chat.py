"""Plan future SWE-Review-Chat pilots without enabling expensive analysis."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot50", "pilot300", "production"), required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--execute-expensive", action="store_true", help="Reserved for the later gated runner")
    args = ap.parse_args()
    if args.execute_expensive:
        raise SystemExit("Expensive SWE-Review-Chat analysis is not enabled in the cheap-audit phase.")
    print(json.dumps({"mode": args.mode, "input": str(args.input), "status": "planned", "expensive_analysis": False}, indent=2))

if __name__ == "__main__": main()
