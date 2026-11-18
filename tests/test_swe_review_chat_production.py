from dataset.config import ALL_FEATURES
from dataset.swe_review_chat_production import (
    ProductionPaths, cheap_eligibility, production_cache_entry_valid,
    normalize_repository_language, normalize_repository_languages, production_failure_reusable,
    select_production_manifests, validate_production_row, validate_startup,
)


def candidate(i, label="LOWER_RISK", files=None, language="Python"):
    row = {"repo_full_name": f"org/repo-{i}", "pr_number": i, "label": int(label == "HIGHER_RISK"),
           "label_name": label, "provenance": [{"verdict": "verified"}], "language": language}
    if files is not None: row["changed_files"] = files
    return row


def full_row(i=1):
    return {"identity": f"org/repo-{i}#{i}", "repo_full_name": f"org/repo-{i}", "pr_number": i,
            "language": "Python", "label": 0, "label_name": "LOWER_RISK",
            "label_provenance": [{"verdict": "approved"}], "provenance": {"full_analysis": True},
            "analysis_mode": "full", "features": {f: 1.0 for f in ALL_FEATURES}}


def test_production_namespace_is_separate_and_complete(tmp_path):
    p = ProductionPaths(tmp_path / "production_v1"); p.prepare()
    assert p.root.name == "production_v1"
    assert all((p.root / n).exists() for n in ("cohorts", "manifests", "ledgers", "checkpoints", "reports"))


def test_cheap_source_eligibility_overrides_repository_language():
    ok, reason, _ = cheap_eligibility(candidate(1, files=["README.md"]), {"org/repo-1": "Python"})
    assert not ok and reason == "NO_SUPPORTED_SOURCE_CHANGE"
    ok, _, detail = cheap_eligibility(candidate(2, files=["src/a.py"]), {"org/repo-2": "Python"})
    assert ok and detail["supported_changed_files"] == ["src/a.py"]


def test_selection_is_deterministic_and_enriched():
    rows = [candidate(i, "HIGHER_RISK" if i < 20 else "LOWER_RISK", [f"a{i}.py"]) for i in range(40)]
    a, natural, excluded = select_production_manifests(rows, {r["repo_full_name"]: "Python" for r in rows}, target=10)
    b, _, _ = select_production_manifests(rows, {r["repo_full_name"]: "Python" for r in rows}, target=10)
    assert [r["identity"] for r in a] == [r["identity"] for r in b]
    assert sum(r["label_name"] == "HIGHER_RISK" for r in a) == 4
    assert not ({r["identity"] for r in a} & {r["identity"] for r in natural})
    assert not excluded


def test_strict_cache_and_row_contracts():
    row = full_row()
    assert validate_production_row(row)[0]
    assert not validate_production_row({**row, "features": {**row["features"], "label": 1}})[0]
    entry = {"identity": row["identity"], "status": "ACCEPTED", "analysis_mode": "full", "features": row["features"]}
    assert production_cache_entry_valid(entry, row["identity"])
    assert not production_failure_reusable({"status": "CHECKOUT_TIMEOUT", "production_version": "production-v1"})
    assert production_failure_reusable({"status": "NO_SUPPORTED_SOURCE_CHANGE", "production_version": "production-v1"})


def test_real_repository_language_summary_shape_and_fallbacks():
    payload = {"repositories": {"a/r": {"mapped_language": "Python", "primary_language": "Python"},
                                "b/r": {"primary_language": "Rust"}, "c/r": "C++", "d/r": None,
                                "e/r": {"language": "unknown"}}}
    assert normalize_repository_language(payload["repositories"]["a/r"]) == "Python"
    assert normalize_repository_language(payload["repositories"]["b/r"]) == "Rust"
    assert normalize_repository_language(payload["repositories"]["c/r"]) == "C++"
    assert normalize_repository_language(payload["repositories"]["d/r"]) is None
    assert normalize_repository_languages(payload) == {"a/r": "Python", "b/r": "Rust", "c/r": "C++"}


def test_real_label_row_and_missing_changed_files_are_staged():
    row = {"repo_full_name": "a/r", "pr_number": 42, "label": 1, "label_name": "HIGHER_RISK",
           "label_reason": "human_changes_requested", "provenance": [{"verdict": "changes_requested"}],
           "state": "closed", "merged_at": None}
    ok, reason, detail = cheap_eligibility(row, {"a/r": {"mapped_language": "Python"}})
    assert ok and reason == "ELIGIBLE" and detail["changed_files_known"] is False


def test_unknown_and_malformed_language_never_become_hash_keys():
    for value in (None, {}, [], {"mapped_language": None}, {"mapped_language": "Kotlin"}):
        row = candidate(100, files=["a.py"]); row.pop("language")
        ok, reason, _ = cheap_eligibility(row, {"org/repo-100": value})
        assert not ok and reason == "UNKNOWN_LANGUAGE"


def test_selection_has_reserve_and_excludes_natural_holdout():
    rows = [candidate(i, "HIGHER_RISK" if i < 30 else "LOWER_RISK", [f"a{i}.py"], "Python") for i in range(60)]
    dev, natural, _ = select_production_manifests(rows, {r["repo_full_name"]: {"mapped_language": "Python"} for r in rows}, target=10)
    assert len(dev) == 10 and len(natural) > 0
    assert not ({r["identity"] for r in dev} & {r["identity"] for r in natural})
    assert len({f"{r['repo_full_name']}#{r['pr_number']}" for r in rows}) == 60


def test_selection_spreads_each_label_across_repositories_before_reusing_one():
    rows = []
    # One repository has a much larger candidate pool than all of the others.
    # A valid development selector must still take the broad one-per-repo pass.
    for label in ("HIGHER_RISK", "LOWER_RISK"):
        for i in range(8):
            repo = "org/dominant" if i < 2 else f"org/{label.lower()}-{i}"
            rows.append({"repo_full_name": repo, "pr_number": 1000 * (label == "HIGHER_RISK") + i,
                         "label": int(label == "HIGHER_RISK"), "label_name": label,
                         "provenance": [{"verdict": "verified"}], "language": "Python",
                         "changed_files": [f"src/{i}.py"]})
    languages = {row["repo_full_name"]: "Python" for row in rows}
    selected, _, _ = select_production_manifests(rows, languages, target=10)
    assert sum(row["label_name"] == "HIGHER_RISK" for row in selected) == 4
    assert sum(row["label_name"] == "LOWER_RISK" for row in selected) == 6
    counts = {}
    for row in selected:
        counts[row["repo_full_name"]] = counts.get(row["repo_full_name"], 0) + 1
    assert max(counts.values()) <= 2


def test_target_specific_final_artifacts_do_not_claim_3000_rows():
    p = ProductionPaths(__import__("pathlib").Path("x") / "production_v1", target=1000)
    assert p.final_csv.name == "final_dataset_1000.csv"
    assert p.final_parquet.name == "final_dataset_1000.parquet"


def test_startup_validation_fails_before_network_on_missing_artifacts(tmp_path):
    try:
        validate_startup(tmp_path, target=3000)
    except RuntimeError as exc:
        assert "missing candidate label ledger" in str(exc)
    else:
        raise AssertionError("startup validation unexpectedly passed")
