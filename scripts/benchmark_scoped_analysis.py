"""Local synthetic scaling benchmark; never contacts GitHub."""
from __future__ import annotations
import argparse, json, shutil, tempfile, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyzer.scoped_analysis import build_repository_index, clear_index_cache, select_dependency_neighborhood

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", nargs="*", type=int, default=[100, 3000, 15000]); args = ap.parse_args()
    rows = []
    for size in args.sizes:
        root = Path(tempfile.mkdtemp(prefix=f"swe_scope_{size}_"))
        try:
            (root / "changed.py").write_text("from relevant import run\ndef entry(): return run()\n", encoding="utf-8")
            (root / "relevant.py").write_text("def run(): return 1\n", encoding="utf-8")
            for i in range(max(0, size - 2)): (root / f"unrelated_{i}.py").write_text("def noise(): return 0\n", encoding="utf-8")
            clear_index_cache(); started = time.perf_counter(); index = build_repository_index(root, str(size), lazy=True); selection = select_dependency_neighborhood(index, ["changed.py"]); elapsed = round((time.perf_counter()-started)*1000, 3)
            rows.append({"repo_files": size, "indexed_files": index.indexed_files, "parsed_files": len(selection.files), "graph_nodes": len(selection.files), "runtime_ms": elapsed})
        finally: shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(rows, indent=2))

if __name__ == "__main__": main()
