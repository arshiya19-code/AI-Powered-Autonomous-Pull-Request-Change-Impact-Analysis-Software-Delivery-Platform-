# Phase 8A — AIDev Dataset Feasibility & Feature-Mapping Audit

**Status:** Audit complete. No training pipeline implemented.  
**Date:** 2026-09-01  
**Analyst:** Antigravity (Phase 8A)

---

## A. Dataset Identity

| Field | Value |
|---|---|
| **Official name** | AIDev |
| **Current version** | v4 (AIDev-2.7M) |
| **Hosted at** | [huggingface.co/datasets/hao-li/AIDev](https://huggingface.co/datasets/hao-li/AIDev) |
| **GitHub replication repo** | [github.com/SAILResearch/AI_Teammates_in_SE3](https://github.com/SAILResearch/AI_Teammates_in_SE3) |
| **Associated papers** | arXiv:2507.15003 ("AIDev: Studying AI Coding Agents on GitHub") and arXiv:2602.09185 ("The Rise of AI Teammates in SE 3.0") |
| **Dataset license** | CC BY 4.0 |
| **Source-repo license** | Each repository retains its own original license. CC-BY-4.0 governs the dataset metadata, not the underlying code/patches. |
| **Cutoff date** | November 2025 |
| **Total PRs** | 2,743,854 |
| **Total repositories** | 327,477 |
| **Human developers** | 159,056 |
| **AI agents tracked** | OpenAI Codex, GitHub Copilot, Cursor, Google Jules, Devin, Claude Code |
| **File format** | Parquet (all tables) |
| **Size on disk** | Varies per table; `pr_commit_details` ~1.3 GB; `all_pull_request` ~1-2 GB |

> [!IMPORTANT]
> The 2.7M figure (`all_pull_request`) includes **both** agent-authored PRs
> **and** all PRs from the same repositories (for control/comparison).
> The curated `pull_request` subset is restricted to repositories with **> 100 stars**.
> The distinction matters for label quality and representativeness.

---

## B. Schema

All tables are available as Parquet files via the Hugging Face `datasets` streaming API or direct download. The schema is confirmed from the Croissant metadata embedded in the dataset card.

### B.1 `all_pull_request` (full PR universe)

| Field | Type | Notes |
|---|---|---|
| `id` | Int64 | **Primary key** (GitHub PR ID) |
| `number` | Int64 | PR number within repository |
| `title` | Text | PR title |
| `body` | Text | PR description body |
| `user` | Text | Author login |
| `user_id` | Int64 | Author GitHub user ID |
| `state` | Text | `open`, `closed` |
| `created_at` | Text (ISO8601) | PR creation timestamp |
| `closed_at` | Text (ISO8601) | PR close timestamp (nullable) |
| `merged_at` | Text (ISO8601) | Merge timestamp (nullable; null = not merged) |
| `repo_url` | Text | GitHub repository URL |
| `repo_id` | Int64 | **Foreign key -> `all_repository.id`** |
| `html_url` | Text | GitHub PR web URL |
| `agent` | Text | Agent name (null = human-authored PR) |

> [!NOTE]
> `merged_at IS NOT NULL` is the primary signal for a merged PR.
> `state = 'closed' AND merged_at IS NULL` means rejected/abandoned.
> There is **no** `base_sha` or `head_sha` in this table.

### B.2 `pull_request` (popular repos, > 100 stars -- curated subset)

Identical schema to `all_pull_request`. This is the recommended starting point for enriched analysis because it is the subset for which `pr_commits`, `pr_commit_details`, `pr_reviews`, etc. are populated.

### B.3 `all_repository`

| Field | Type | Notes |
|---|---|---|
| `id` | Int64 | **Primary key** |
| `url` | Text | GitHub repository URL |
| `full_name` | Text | `owner/repo` |
| `license` | Text | SPDX license identifier (nullable) |
| `language` | Text | GitHub-reported primary language (nullable) |
| `is_forked` | Boolean | Whether the repo is a fork |
| `forks` | Float64 | Fork count |
| `stars` | Float64 | Star count |

### B.4 `repository` (curated subset, > 100 stars)

Identical schema to `all_repository`. Stars and forks are Int64 here (not Float64).

### B.5 `pr_commits`

| Field | Type | Notes |
|---|---|---|
| `sha` | Text | Commit SHA (**primary key** within a PR) |
| `pr_id` | Int64 | **Foreign key -> `pull_request.id`** |
| `author` | Text | Commit author login |
| `committer` | Text | Commit committer login |
| `message` | Text | Commit message |

> [!WARNING]
> `pr_commits` does **not** contain `parent_sha`, `base_commit`, or `head_commit`.
> The base/head commit SHAs are not directly available in any AIDev table.
> Repository checkout requires deriving the parent from the commit graph via the GitHub API or `git log`.

### B.6 `pr_commit_details`

This is the most important table for our purpose. It is file-level detail per commit.

| Field | Type | Notes |
|---|---|---|
| `sha` | Text | Commit SHA (join to `pr_commits.sha`) |
| `pr_id` | Int64 | **Foreign key -> `pull_request.id`** |
| `author` | Text | Commit author |
| `committer` | Text | Commit committer |
| `message` | Text | Commit message |
| `commit_stats_total` | Int64 | Total lines changed in commit |
| `commit_stats_additions` | Int64 | Total additions in commit |
| `commit_stats_deletions` | Int64 | Total deletions in commit |
| `filename` | Text | **File path** (one row per file per commit) |
| `status` | Text | `added`, `modified`, `removed`, `renamed` |
| `additions` | Float64 | Line additions in this file |
| `deletions` | Float64 | Line deletions in this file |
| `changes` | Float64 | Total line changes in this file |
| `patch` | Text | **Unified diff patch text** (nullable -- can be missing for large files) |

> [!IMPORTANT]
> `patch` is the primary input to our analyzer pipeline. It can be:
> - **Present and complete** -- most common for small/medium files
> - **Truncated** -- GitHub API truncates patches for files > ~1 MB
> - **Absent (null)** -- binary files, very large files, or API failure
>
> A fallback fetch via `https://github.com/{owner}/{repo}/commit/{sha}.diff` is required for truncated cases.

### B.7 `pr_reviews`

| Field | Type | Notes |
|---|---|---|
| `id` | Int64 | Review ID |
| `pr_id` | Int64 | **Foreign key -> `pull_request.id`** |
| `user` | Text | Reviewer login |
| `user_type` | Text | `User`, `Bot` |
| `state` | Text | `APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`, `DISMISSED` |
| `submitted_at` | Text | Review submission timestamp |
| `body` | Text | Review comment body |

### B.8 `pr_review_comments` (pr_review_comments_v2 in Croissant spec)

| Field | Type | Notes |
|---|---|---|
| `id` | Int64 | Comment ID |
| `pull_request_review_id` | Int64 | Foreign key -> review |
| `user` | Text | Commenter login |
| `user_type` | Text | `User`, `Bot` |
| `diff_hunk` | Text | Code context of the comment |
| `path` | Text | File path commented on |
| `position` | Int64 | Position in diff |
| `original_position` | Int64 | Original position |
| `commit_id` | Text | Commit SHA the comment is on |
| `original_commit_id` | Text | Original commit SHA |
| `body` | Text | Comment text |
| `pull_request_url` | Text | PR URL |
| `created_at` | Text | Comment creation timestamp |
| `updated_at` | Text | Comment last update |
| `in_reply_to_id` | Float64 | Reply thread ID (nullable) |

### B.9 `pr_comments`

| Field | Type | Notes |
|---|---|---|
| `id` | Int64 | Comment ID |
| `pr_id` | Int64 | **Foreign key -> `pull_request.id`** |
| `user` | Text | Commenter login |
| `user_id` | Int64 | Commenter user ID |
| `user_type` | Text | `User`, `Bot` |
| `created_at` | Text | Timestamp |
| `body` | Text | Comment body text |

### B.10 `pr_task_type` (annotation table)

| Field | Type | Notes |
|---|---|---|
| `pr_id` | Int64 | **Foreign key -> `pull_request.id`** |
| `task_type` | Text | LLM-classified conventional-commit category |

Observed `task_type` values: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `style`, `perf`, `revert`, `other`.

### B.11 Tables NOT Present in AIDev v4

| Requested | Status |
|---|---|
| `pr_timeline` | **NOT CONFIRMED in schema** -- not listed in Croissant metadata |
| `all_user` | Present (user metadata: followers, following, created_at) |
| CI/check-run status | **ABSENT** -- no `check_runs`, `workflow_runs`, or `ci_status` table exists |

> [!CAUTION]
> **No CI/CD outcome data is available in AIDev v4.**
> There are no `check_runs`, `workflow_run`, test-failure, or GitHub Actions result fields.
> CI-based labels are **not constructible** from AIDev data alone.

---

## C. Join Strategy

The canonical join chain for our pipeline:

```
pull_request
    |  JOIN ON pull_request.id = pr_commits.pr_id
    v
pr_commits           (one row per commit; gives SHA)
    |  JOIN ON pr_commits.sha = pr_commit_details.sha
    v
pr_commit_details    (one row per file per commit; gives patch, additions, deletions)
    |  JOIN ON pull_request.repo_id = repository.id
    v
repository           (gives language, license, full_name)
    |  JOIN ON pull_request.id = pr_reviews.pr_id
    v
pr_reviews           (gives APPROVED / CHANGES_REQUESTED signals)
    |  JOIN ON pull_request.id = pr_task_type.pr_id
    v
pr_task_type         (gives feat / fix / revert / ... classification)
```

**Aggregation required:** Because `pr_commit_details` is file x commit level and our analyzer consumes a whole-PR unified diff, all file patches for a given `pr_id` must be concatenated (in commit order) to reconstruct the PR-level diff.

**Reconstruction of repository snapshot:** AIDev does **not** store base commit SHAs. Two approaches:

1. **Patch-only mode** -- apply the `patch` text without full repository context. Enables Category A and B features only.
2. **Repository checkout** -- clone the repo and check out the base commit. Enables all 42 features. Requires:
   - `repository.full_name` -> `git clone https://github.com/{full_name}`
   - First commit SHA from `pr_commits.sha` -> `git log` to find the merge base
   - Controlled sampling -- do NOT clone thousands of repos.

---

## D. License Strategy

> [!IMPORTANT]
> **CC-BY-4.0 governs the AIDev dataset metadata only.**
> The patches embedded in `pr_commit_details.patch` are derived from original source
> repositories which retain their own licenses. We cannot treat every patch as CC-BY-4.0
> without verifying the underlying repository license.

The pipeline must accept a **researcher-configured license allowlist**:

```python
# dataset/config.py  (to be created in Phase 8B)
# This list is a researcher-selected convenience filter.
# It does NOT constitute legal advice.
ALLOWED_LICENSES: set[str] = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    # Add others after researcher review:
    # "mpl-2.0",
    # "gpl-2.0",
    # "gpl-3.0",
}

ALLOW_UNKNOWN_LICENSES: bool = False  # exclude null/other by default
```

Matching must be **case-insensitive** and handle common SPDX variants (e.g., `MIT`, `MIT License`).
GPL / AGPL / LGPL repos require separate researcher review before inclusion.

---

## E. Language Distribution

The `repository.language` field reflects GitHub's primary language detection. Exact counts require a streaming scan. From the AIDev research papers and dataset description, the approximate distribution across 327,477 repositories:

| Language | Approx. Proportion | Notes |
|---|---|---|
| JavaScript / TypeScript | ~22-28% | Combined; JS is the single largest |
| Python | ~18-22% | Second largest; most relevant for Phase 8B |
| Java | ~8-12% | Significant presence |
| Go | ~5-8% | Growing share |
| C / C++ | ~4-7% | Complex toolchain for checkout |
| Ruby | ~3-5% | Rails-heavy repositories |
| PHP | ~2-4% | -- |
| Rust | ~2-3% | Small but growing |
| Other / null | ~15-20% | Mixed/unknown language |

> [!NOTE]
> Exact counts are only obtainable by streaming the `repository` table.
> The Phase 8B ingestion script must emit a language frequency table as its first output.

**Python-only initial experiment estimate:** Applying language = `Python` + MIT/Apache allowlist + curated `pull_request` subset (> 100 stars), an estimated **15,000-40,000 PRs** are eligible. After requiring patch availability and at least one review, the usable pool drops to ~3,500-7,000. The **initial experiment targets 300 PRs** from this pool.

---

## F. Phase 6 Feature Mapping

The complete Phase 6 feature vector is the union of Phase 5 features (from `ImpactAnalysisResult._build_features` in `change_analyzer.py:1403-1430`) and Phase 6 additions (from `ImpactScorer._build_features` in `impact_scorer.py:772-808`).

### Legend

- **Cat A** = Directly available from AIDev fields (no code required)
- **Cat B** = Computed by our analyzer from `patch` text (no repo checkout required)
- **Cat C** = Requires repository checkout + full pipeline (AST -> graph -> analysis)

### Phase 5 Base Features (20 features, `change_analyzer.py`)

| # | Feature Name | Type | Direct AIDev Field | AIDev Table | Category | Availability |
|---|---|---|---|---|---|---|
| 1 | `files_changed` | float | `filename` (distinct count per pr_id) | `pr_commit_details` | A | AVAILABLE |
| 2 | `lines_added` | float | `additions` (sum per pr_id) | `pr_commit_details` | A | AVAILABLE |
| 3 | `lines_deleted` | float | `deletions` (sum per pr_id) | `pr_commit_details` | A | AVAILABLE |
| 4 | `total_lines_changed` | float | `additions + deletions` | `pr_commit_details` | A | AVAILABLE |
| 5 | `functions_changed` | float | -- | `pr_commit_details.patch` | B | REQUIRES PATCH |
| 6 | `classes_changed` | float | -- | `pr_commit_details.patch` | B | REQUIRES PATCH |
| 7 | `methods_changed` | float | -- | `pr_commit_details.patch` | B | REQUIRES PATCH |
| 8 | `dependencies_added` | float | -- | `pr_commit_details.patch` | B | REQUIRES PATCH |
| 9 | `dependencies_removed` | float | -- | `pr_commit_details.patch` | B | REQUIRES PATCH |
| 10 | `dependencies_changed` | float | -- | `pr_commit_details.patch` | B | REQUIRES PATCH |
| 11 | `direct_nodes_changed` | float | -- | Patch + graph | C | REQUIRES CHECKOUT |
| 12 | `impacted_nodes` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |
| 13 | `impacted_modules` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |
| 14 | `impacted_functions` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |
| 15 | `impacted_classes` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |
| 16 | `impacted_methods` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |
| 17 | `maximum_impact_depth` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |
| 18 | `impact_ratio` | float | -- | impacted / total_nodes | C | REQUIRES CHECKOUT |
| 19 | `dependency_change_ratio` | float | -- | dep_changes / direct+1 | B/C | PARTIAL |
| 20 | `average_impact_depth` | float | -- | Graph traversal | C | REQUIRES CHECKOUT |

### Phase 6 Additional Features (22 features, `impact_scorer.py`)

| # | Feature Name | Type | Category | Availability |
|---|---|---|---|---|
| 21 | `changed_modules` | float | C | REQUIRES CHECKOUT |
| 22 | `cross_module_impact_count` | float | C | REQUIRES CHECKOUT |
| 23 | `cross_module_ratio` | float | C | REQUIRES CHECKOUT |
| 24 | `average_changed_node_degree` | float | C | REQUIRES CHECKOUT |
| 25 | `maximum_changed_node_degree` | float | C | REQUIRES CHECKOUT |
| 26 | `average_changed_node_dependents` | float | C | REQUIRES CHECKOUT |
| 27 | `maximum_changed_node_dependents` | float | C | REQUIRES CHECKOUT |
| 28 | `average_changed_node_dependencies` | float | C | REQUIRES CHECKOUT |
| 29 | `maximum_changed_node_dependencies` | float | C | REQUIRES CHECKOUT |
| 30 | `import_propagation_count` | float | C | REQUIRES CHECKOUT |
| 31 | `from_import_propagation_count` | float | C | REQUIRES CHECKOUT |
| 32 | `call_propagation_count` | float | C | REQUIRES CHECKOUT |
| 33 | `inheritance_propagation_count` | float | C | REQUIRES CHECKOUT |
| 34 | `decoration_propagation_count` | float | C | REQUIRES CHECKOUT |
| 35 | `total_propagation_edges` | float | C | REQUIRES CHECKOUT |
| 36 | `critical_node_count` | float | C | REQUIRES CHECKOUT |
| 37 | `nodes_at_depth_0` | float | C | REQUIRES CHECKOUT |
| 38 | `nodes_at_depth_1` | float | C | REQUIRES CHECKOUT |
| 39 | `nodes_at_depth_2` | float | C | REQUIRES CHECKOUT |
| 40 | `nodes_at_depth_3_plus` | float | C | REQUIRES CHECKOUT |
| 41 | `impact_score` | float | C | REQUIRES CHECKOUT |
| 42 | `impact_level_numeric` | float | C | REQUIRES CHECKOUT |

**Total features: 42** (20 Phase 5 + 22 Phase 6)

**Summary by category:**

| Category | Count | Description |
|---|---|---|
| A -- Directly from AIDev | 4 | `files_changed`, `lines_added`, `lines_deleted`, `total_lines_changed` |
| B -- Patch-only (no checkout) | 6 | `functions_changed`, `classes_changed`, `methods_changed`, `dependencies_added/removed/changed` |
| B/C -- Partial | 1 | `dependency_change_ratio` |
| C -- Requires repo checkout | 31 | All graph-based, transitive, connectivity, propagation, depth features |

> [!IMPORTANT]
> **31 of 42 features (74%) require a full repository checkout at the base commit.**
> A patch-only run with only 10 Cat A+B features significantly reduces model quality.
> A proper initial experiment requires controlled checkout of ~50 unique repositories.

---

## G. Label Feasibility

| Label Candidate | AIDev Evidence | Feasibility |
|---|---|---|
| PR merged (`merged_at IS NOT NULL`) | `pull_request.merged_at` | AVAILABLE |
| PR closed without merge | `state`, `merged_at` | AVAILABLE |
| Review required changes (`CHANGES_REQUESTED`) | `pr_reviews.state` | AVAILABLE |
| Review approved without changes | `pr_reviews.state` | REQUIRES DERIVATION |
| Revert PR (`task_type = 'revert'`) | `pr_task_type.task_type` | AVAILABLE (very rare, ~16-31 per major agent) |
| CI/test failure | No CI data in AIDev | NOT AVAILABLE |
| Post-merge bug-fix PR | Temporal self-join possible | REQUIRES DERIVATION (complex) |
| Rollback | Only revert proxy | WEAK PROXY ONLY |
| Deployment outcome | Not in AIDev | NOT AVAILABLE |

---

## H. Proposed Target Label

### Recommended: Label B -- "Review Friction" (`CHANGES_REQUESTED`)

**Definition:**

```python
# PR is labeled HIGH RISK (1) if it received at least one CHANGES_REQUESTED review.
# PR is labeled LOW RISK (0) if only APPROVED or COMMENTED reviews (or no reviews),
# and was not a revert.
label = 1 if any(r.state == "CHANGES_REQUESTED" for r in pr.reviews) else 0
```

**Justification:**
- `CHANGES_REQUESTED` is the strongest available signal that a reviewer identified a problem.
- It is the closest proxy to "deployment risk" without CI data.
- It reflects pre-merge information about the change's quality.
- It is not leakage as long as reviewer verdict is used as the label, not as a feature.

**Known weaknesses:**
- `CHANGES_REQUESTED` != deployment failure; some requests are stylistic.
- Some high-risk PRs are merged without review (labeled 0 incorrectly).
- Expected class imbalance: ~70-80% label=0, ~20-30% label=1 for agent-authored PRs.

### Secondary: Label A -- "Merged vs. Closed-Without-Merge"

```python
label = 0 if merged_at is not None else 1
```

**Caveat:** This is an *acceptance* label, not a *risk* label. Use as auxiliary signal only.

### Tertiary (deferred): Label C -- "Followed by Revert/Fix PR"

Requires temporal self-join across PRs in the same repository. Deferred to Phase 8C.

---

## I. Leakage Analysis

### MODEL INPUT FEATURES (safe to use as inputs)

Features observable at diff time, before outcome is known:

- `files_changed`, `lines_added`, `lines_deleted`, `total_lines_changed`
- `functions_changed`, `classes_changed`, `methods_changed`
- `dependencies_added`, `dependencies_removed`, `dependencies_changed`
- All graph-based features (computed on repo at base commit + patch)
- `impact_score`, `impact_level_numeric`

### LABEL-ONLY / POST-OUTCOME FEATURES (must NOT be model inputs)

| Field | Table | Reason |
|---|---|---|
| `merged_at` | `pull_request` | Outcome timestamp -- defines label |
| `closed_at` | `pull_request` | Outcome timestamp |
| `state` (when `closed`) | `pull_request` | Reveals final outcome |
| `pr_reviews.state` | `pr_reviews` | Review verdict IS the target label |
| `pr_reviews.submitted_at` | `pr_reviews` | Post-PR-open event |
| `pr_task_type.task_type = 'revert'` | `pr_task_type` | Post-outcome annotation |
| Any data from a follow-up PR | Multiple tables | Future event |

> [!CAUTION]
> `merged_at`, `closed_at`, `state`, and all `pr_reviews.*` fields must be used
> **ONLY for label construction**, never as model inputs.
> This must be enforced by a strict feature/label split in Phase 8B ingestion code.

---

## J. Initial Experiment Subset

### Filtering Criteria (deterministic, reproducible)

```python
SUBSET_FILTERS = {
    # 1. Use curated subset only (popular, well-maintained repos)
    "table": "pull_request",          # NOT all_pull_request

    # 2. Language filter (Phase 7 has Python adapter only)
    "language": "Python",             # repository.language == "Python"

    # 3. License allowlist (researcher-configurable)
    "allowed_licenses": {
        "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"
    },

    # 4. Exclude forks
    "is_forked": False,

    # 5. Patch availability required
    "patch_required": True,

    # 6. Review availability required (for Label B)
    "review_required": True,

    # 7. PR must be closed
    "pr_state": "closed",

    # 8. Exclude revert task type
    "exclude_task_types": {"revert"},

    # 9. Deterministic sampling
    "sample_size": 300,
    "random_seed": 42,
}
```

### Expected yield estimation

| Filter stage | Estimated pool remaining |
|---|---|
| `pull_request` curated subset | ~150,000-200,000 PRs |
| Python language | ~25,000-40,000 |
| MIT/Apache/BSD licenses | ~15,000-25,000 |
| Not forked | ~12,000-20,000 |
| Has patch data | ~10,000-16,000 |
| Has at least one review | ~4,000-8,000 |
| State = closed | ~3,500-7,000 |
| **Target sample** | **300 PRs (seed=42)** |

---

## K. Computational Strategy

> [!IMPORTANT]
> `pr_commit_details.parquet` is approximately 1.3 GB.
> Loading these entirely into `pandas` on a 16 GB machine leaves no headroom for the analyzer.

### Strategy: Streaming + Column-Pruned + Chunked (Polars)

```python
# Pseudocode -- actual implementation deferred to Phase 8B

import polars as pl

# Step 1: Load repository metadata (small table, load fully)
repo_df = pl.read_parquet(
    "hf://datasets/hao-li/AIDev/repository/train/*.parquet",
    columns=["id", "full_name", "language", "license", "is_forked", "stars"],
)
allowed_repos = repo_df.filter(
    (pl.col("language") == "Python") &
    (pl.col("license").str.to_lowercase().is_in(ALLOWED_LICENSES)) &
    (~pl.col("is_forked"))
)

# Step 2: Lazy scan pull_request with column pruning
pr_df = pl.scan_parquet(
    "hf://datasets/hao-li/AIDev/pull_request/train/*.parquet",
    columns=["id", "repo_id", "state", "merged_at", "created_at", "agent"],
).filter(
    pl.col("repo_id").is_in(allowed_repos["id"]) &
    (pl.col("state") == "closed")
).collect()

# Step 3: Filter reviews (require at least one review per PR)
reviews_df = pl.scan_parquet(
    "hf://datasets/hao-li/AIDev/pr_reviews/train/*.parquet",
    columns=["pr_id", "state"],
).filter(
    pl.col("pr_id").is_in(pr_df["id"])
).collect()

# Step 4: Sample 300 PRs deterministically
sampled_pr_ids = (
    pr_df
    .filter(pl.col("id").is_in(reviews_df["pr_id"]))
    .sample(n=300, seed=42)["id"]
    .to_list()
)

# Step 5: Stream pr_commit_details for sampled PRs ONLY
commit_details_df = pl.scan_parquet(
    "hf://datasets/hao-li/AIDev/pr_commit_details/train/*.parquet",
    columns=["pr_id", "sha", "filename", "status", "additions", "deletions", "patch"],
).filter(
    pl.col("pr_id").is_in(sampled_pr_ids)
).collect()
```

**Key principles:**
- Use **Polars `scan_parquet`** (lazy evaluation) for large tables -- never `read_parquet` for multi-GB files.
- **Column prune** immediately -- only request needed columns.
- **Filter before collect** -- push the `pr_id in sampled_pr_ids` predicate into the scan.
- Process PRs **one at a time** through the analyzer.
- **Repository checkout** is sequential and temporary: `git clone --depth=1`, process, `rm -rf`. Never keep > 5 repos on disk simultaneously.

**Preferred libraries (no new dependencies beyond likely already present):**
- `polars` -- lazy Parquet scanning, no pandas overhead
- `pyarrow` -- transitive dependency of polars
- `huggingface_hub` -- for streaming access
- Standard `subprocess` for `git clone`

---

## L. Risks and Limitations

| Risk | Severity | Mitigation |
|---|---|---|
| **Missing patches** (`patch = null`) | HIGH | Skip file; record null-patch rate; fetch raw diff from GitHub API as fallback |
| **Truncated patches** (large files) | MEDIUM | Detect truncation (patch ends mid-hunk); fetch from `{repo_url}/commit/{sha}.diff` |
| **Repository checkout cost** | HIGH | Limit Phase 8B to <=50 unique repos for 300 PRs; use `git clone --depth` |
| **Source repository license uncertainty** | HIGH | Apply allowlist strictly; log excluded repos for researcher review |
| **No CI outcome data** | HIGH | Label B (`CHANGES_REQUESTED`) is the best available proxy; acknowledge limitation |
| **Class imbalance** | HIGH | Expect ~70-80% label=0; use class weights in CatBoost |
| **Label quality -- reviewer subjectivity** | MEDIUM | `CHANGES_REQUESTED` varies by reviewer; some repos have no reviews at all |
| **Unsupported languages** | MEDIUM | Filter to Python only; zero-pad graph features for non-Python (future) |
| **Patch parsing failures** | MEDIUM | AI-generated patches may have unusual formatting; validate before AST analysis |
| **Target leakage** | CRITICAL | Enforce strict feature/label split; never pass `merged_at`, `state`, or `pr_reviews.state` as features |
| **Graph construction for large repos** | MEDIUM | Cap graph node count at 10,000; log and skip oversized repos |
| **Revert PRs distorting labels** | LOW | Exclude `task_type = 'revert'` from initial experiment |
| **Agent-human confound** | MEDIUM | Record `pr.agent` field; track agent vs. human split but do not exclude human PRs |

---

## Summary Answers to Phase 8A Questions

| # | Question | Answer |
|---|---|---|
| 1 | Exact AIDev version | v4 (AIDev-2.7M), cutoff November 2025 |
| 2 | Tables inspected | `pull_request`, `all_pull_request`, `repository`, `all_repository`, `pr_commits`, `pr_commit_details`, `pr_reviews`, `pr_review_comments`, `pr_comments`, `pr_task_type`, `all_user` |
| 3 | Relevant fields | See Section B; key: `merged_at`, `agent`, `language`, `license`, `patch`, `additions`, `deletions`, `pr_reviews.state` |
| 4 | License implications | CC-BY-4.0 on metadata only; repo code patches retain original license; configure allowlist |
| 5 | Languages available | JS/TS ~25%, Python ~20%, Java ~10%, Go ~6%, C/C++ ~5%, others; exact counts need streaming scan |
| 6 | Usable PRs after filters | ~3,500-7,000 Python/licensed/reviewed/closed; target 300 for initial experiment |
| 7 | Phase 6 feature mapping | 42 features total: 4 Cat A, 6 Cat B, 1 Cat B/C, 31 Cat C |
| 8 | Features requiring checkout | 31 of 42 (all graph-based, transitive, connectivity, propagation, depth features) |
| 9 | Available outcome labels | Merged vs. closed (yes), CHANGES_REQUESTED (yes), revert task type (yes, rare); NO CI data |
| 10 | Recommended label | `1` if any `pr_reviews.state = 'CHANGES_REQUESTED'` else `0` |
| 11 | Potential leakage | `merged_at`, `closed_at`, `state`, all `pr_reviews.*`, any post-merge events |
| 12 | Initial dataset size | 300 PRs, seed=42, from Python/MIT-Apache/non-forked/reviewed/closed pool |
| 13 | Multi-language strategy | Phase 8B: Python only. Phase 8C+: Java/JS/TS as adapters are built |
| 14 | RAM/processing strategy | Polars lazy scan + column pruning; never load full 1.3 GB table; sequential per-PR processing |
| 15 | Files created | `docs/aidev_dataset_audit.md` |
| 16 | Tests | None in Phase 8A (no code created); Phase 8B will add ingestion/schema validation tests |
| 17 | Unresolved uncertainties | (a) exact language distribution; (b) patch truncation rate unknown until sampled; (c) whether 300 PRs with Cat C features is achievable in 16 GB RAM; (d) whether `pr_task_type` covers the entire curated `pull_request` subset |

---

## Conclusion

**Can real AIDev PRs be reliably transformed into our existing Phase 6 feature vectors with a defensible real-world risk label?**

**Conditionally yes**, with these constraints:

1. **31 of 42 Phase 6 features require repository checkout.** A patch-only run produces only 10 reliable features, insufficient for the model.
2. **The recommended label (`CHANGES_REQUESTED`)** is defensible but is a *review friction* label, not a *deployment outcome* label. This distinction must be stated in any model card.
3. **No CI/CD data is available.** This is a fundamental limitation of AIDev v4.
4. **Python-only initial experiment is feasible** with ~3,500-7,000 candidate PRs and a 300-PR controlled first run.
5. **Memory and compute are manageable** with Polars lazy loading and sequential checkout.

**Phase 8B recommendation:** Design the ingestion pipeline to support both patch-only mode (10 features, no checkout) and full-checkout mode (42 features). Run patch-only first to validate the label distribution before committing to the checkout pipeline.
