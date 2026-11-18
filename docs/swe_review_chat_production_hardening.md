# SWE-Review-Chat production pipeline hardening

## Bottleneck map

The old path paid repository-size-dependent costs in clone/fetch, checkout,
recursive source enumeration, blob reads, resolver indexing, and graph build.
The hardened path pays network cost once per repository object cache, uses
immutable base/head SHAs, and confines adapter parsing, resolver input, graph,
and impact work to the selected closure. Path indexing remains revision-keyed;
semantic parse results are not shared across revisions.

## Acquisition architecture

`data/processed/swe_review_chat/repos/<stable-id>/repo.git` is a bare cache,
protected by a repository lock. It is validated with `rev-parse
--is-bare-repository`, fetched with `--no-tags` and `--filter=blob:none` when
supported, and repaired only when invalid. Exact SHA, PR-ref, and fork paths
are attempted without substituting an unrelated commit. Analysis receives an
isolated linked worktree; a worktree failure never deletes a valid object cache.

## Safety and diagnostics

Limits are based on parsed files, graph nodes, closure size, and runtime. The
pipeline reports precise statuses including `BASE_COMMIT_UNAVAILABLE`,
`HEAD_COMMIT_UNAVAILABLE`, `GIT_FETCH_TIMEOUT`, `WORKTREE_FAILED`,
`DEPENDENCY_CLOSURE_LIMIT_EXCEEDED`, `ANALYSIS_RUNTIME_LIMIT_EXCEEDED`, and
`GRAPH_RESOURCE_LIMIT_EXCEEDED`. Diagnostics are kept beside, never inside,
the canonical ordered 42-feature vector.

The local parity utility is `scripts/diagnose_swe_review_chat_parity.py`; the
synthetic benchmark is `scripts/benchmark_scoped_analysis.py`.
