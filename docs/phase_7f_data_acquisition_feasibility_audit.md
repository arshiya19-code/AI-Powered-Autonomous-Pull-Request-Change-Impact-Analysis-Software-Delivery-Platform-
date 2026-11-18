# Phase 7F — Data Acquisition Feasibility Audit

**Audit date:** 2026-09-06  
**Scope:** feasibility only. No repository acquisition, label generation, feature extraction, adapter work, model work, or sampling was performed.

## 1. Executive Verdict

## FEASIBLE WITH CONSTRAINTS

A useful, reproducible **Python-only initial** HIGH-versus-LOW_OBSERVED dataset is technically constructible from a pinned AIDev candidate catalog plus repository history and a frozen GitHub evidence cache. It is not constructible from AIDev alone. The proposed broad multi-language evaluation is **not currently achievable**: Python is the only adapter that has passed the project validation gate, and no count yet demonstrates 50 eligible repository networks for any other language. High-confidence HIGH labels are observable only where a post-merge revert/rollback has explicit, retained linkage to the original PR; LOW_OBSERVED needs a successfully completed 90-day evidence scan, not mere absence in AIDev. Whole-network external validation is achievable after GitHub-derived network mapping and patch/event deduplication are frozen, but not with AIDev's repository name alone.

The usable upstream revision is `c63c8a57a2de34fc03fa83722412824fc4d8753b` (the v4/AIDev-2.7M commit). The live card identifies v4 as 2,743,854 PRs in 327,477 repositories with a November 2025 cutoff, and exposes the full tables plus the enriched >100-star AIDev-pop subset. [AIDev dataset card](https://huggingface.co/datasets/hao-li/AIDev) [pinned file tree](https://huggingface.co/datasets/hao-li/AIDev/tree/c63c8a57a2de34fc03fa83722412824fc4d8753b)

## 2. AIDev Schema Findings

The source's documented normalized tables are `all_pull_request`, `all_repository`, `all_user`, and, for AIDev-pop, `pull_request`, `repository`, `pr_comments`, `pr_reviews`, `pr_review_comments`, `pr_commits`, and `pr_commit_details`. The latter contains file-level paths, stats, and often a patch; it omits large patches when GitHub did not supply them. The full PR table is about 1.51 GB and `pr_commit_details` about 1.33 GB, so the source requires column-pruned, record-batch scans. [AIDev data structure and loading guidance](https://huggingface.co/datasets/hao-li/AIDev)

| Field | Available? | Source table | Reproducible? | Important limitations |
|---|---|---|---|---|
| PR global ID / number | Yes | `all_pull_request`, `pull_request` | Yes, pinned | Number is only unique within repository. |
| Repository ID / URL / owner-name | Yes | PR `repo_id`/`repo_url`; repository `id`/`full_name` | Yes, pinned | URL/name may no longer resolve after rename/deletion. |
| Primary language | Yes | repository `language` | Yes, pinned | GitHub primary repository language, not changed-file language. |
| Created / closed / merged timestamps | Yes | PR tables | Yes, pinned | Use `merged_at`, parsed as UTC; null means no recorded merge. |
| Base SHA | No | — | No from AIDev | Must be captured from provider PR object or derived/verified from Git graph. |
| Final PR-head SHA | Indirect only | `pr_commits.sha` | Conditional | Commit list can identify a candidate last PR commit, but is not a provider guarantee of t0. Capture `head.sha` from GitHub. |
| Merge commit SHA | No direct field | — | No from AIDev | Fetch provider PR object; merge strategy changes meaning of `merge_commit_sha`. |
| Commit history / messages | Partial | `pr_commits`, `pr_commit_details` | Yes, pinned | No parent/base graph. Provider/git history remains necessary. |
| Changed paths / per-file additions/deletions | Yes, enriched subset | `pr_commit_details` | Yes, pinned | File x commit rows require aggregation; absent from full-only universe. |
| Diff/patch | Partial | `pr_commit_details.patch` | Conditional | Missing/truncated large/binary patches; never treat absence as empty change. |
| Changed-line counts | Yes, enriched subset | `pr_commit_details` stats | Yes, pinned | Aggregate carefully to avoid counting a file repeatedly across commits. |
| Fork status | Partial | repository `is_forked` | Yes, pinned | No parent/source/network ID. |
| Upstream/fork network, patch hash, outcome table | No | — | No | Must be enriched and frozen externally. |
| Deleted/unavailable repository | No historical flag | — | Only at acquisition | Record provider lookup/clone outcome, timestamp, and HTTP/git error. |

The exact field names above are confirmed for the current source's published loading/schema material and the local prior schema record; a fresh parquet footer/schema manifest must be captured in the acquisition run. No full source scan or download was performed in this audit.

## 3. Snapshot Reconstruction

For each selected merged candidate, resolve the AIDev PR identity to the provider PR object and freeze `base.sha`, `head.sha`, `merged_at`, `merge_commit_sha`, repository node/ID, URL, and retrieval timestamp. `t0` is the resolved final `head.sha` immediately before merge; it is not the merge commit and cannot be inferred safely from the current default branch. Fetch a bare/full git object store (not a shallow clone), verify that base and head objects exist, compute `base..head`, and materialize the head tree plus the base tree. Perform all 42-feature extraction only from those t0-visible objects. Keep the PR patch only as a cross-check/fallback, not as proof of a complete snapshot.

GitHub's post-merge `merge_commit_sha` is merge-method dependent: merge commit, squashed base commit, or the commit to which the base branch was updated after a rebase. Store both it and the original head SHA. [GitHub pull-request API semantics](https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28)

Failures are exclusion/UNKNOWN cases: renamed/deleted/private repositories, removed or force-pushed objects, inaccessible PRs, absent large-file patches, partial/shallow clones, LFS/submodule gaps, rate limits, and historical rewrites. A provider response must be retained; no reconstruction failure may be represented as a zero-valued feature vector.

## 4. Outcome Evidence

| Evidence type | Source/API | Linkage mechanism | Reproducible? | Reliability | Leakage risk |
|---|---|---|---|---|---|
| Confirmed revert/rollback | Git commit object, PR commit list/diff, PR/timeline reference | Explicit `This reverts commit <sha>` matching merge/head/reachable original commit; preserve commit, diff, timestamp, and PR references | Yes, if raw responses/git objects are cached | High when explicit reference and commit reachability verify | Label-only, post-t0 |
| GitHub PR/commit association | GitHub PR object, commits endpoint, GraphQL/REST timeline | Provider association plus explicit original PR number/URL/SHA | Yes, cached | Medium–high; association alone is not a revert | Label-only |
| Explicit corrective PR | PR title/body/comments/timeline + merged corrective commit/diff | Explicit original PR reference **and** material scope overlap, adjudicated by versioned rule | Yes, cached | Medium | Label-only |
| Commit-message PR reference | Git log / commit API | Repository-qualified `#number` or URL, then resolve and verify time/scope | Yes, cached | Low alone; candidate generation only | Label-only |
| Cross-referenced issue/PR | GitHub issue timeline | `cross-referenced` event pointing to a PR/issue that explicitly identifies original PR | Yes, cached | Medium; not enough without semantic linkage | Label-only |
| Incident/issue | GitHub issue/timeline | Explicit original PR/commit reference and corroborating corrective/revert evidence | Conditional | Low–medium alone | Label-only |
| Same-file later commit | Git history | None sufficient | No qualifying linkage | Unreliable; must not label | Forbidden as outcome criterion |

GitHub timeline events can expose PR/issue cross-references, including the referencing source and timestamp, but they are supporting linkage, not a rollback relationship. [Timeline events](https://docs.github.com/en/rest/issues/timeline) [cross-reference event fields](https://docs.github.com/en/rest/using-the-rest-api/issue-event-types)

**HIGH rule:** accept only a post-merge, in-window reverted/rollback commit with reliable direct linkage to the original merged change. A conventional Git generated revert message and reachable target SHA is sufficient subject to stored verification. A manual rollback needs explicit PR/commit reference plus substantive patch confirmation; otherwise it is MEDIUM, UNKNOWN, or CONFLICTED. **MEDIUM rule:** explicit link plus materially overlapping scope; it never enters the first binary model. Search results, similarity, shared files, or keyword matches are only triage signals and cannot create a label.

## 5. Repository/Fork Network Identity

AIDev's stable repository ID and `is_forked` flag establish individual repositories, not families. GitHub's repository endpoint supplies `parent` for the immediate fork and `source` for the ultimate fork-network source when a repository is a fork. Freeze the source repository node/ID as the initial network ID, retaining parent/source payloads and lookup time. [GitHub repository endpoint](https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28)

This handles provider-declared GitHub forks, but cannot guarantee detection of independently copied repositories, deleted parent records, or non-GitHub mirrors. Add exact patch hashes computed from canonical `base..head` diffs, repeated-commit SHA groups, and a conservative near-duplicate candidate table; do not automatically merge near-duplicates without an auditable rule. Network membership, exact duplicate groups, and related revert/corrective-event groups must all be frozen before split construction. `repo_name` is never a network ID.

## 6. 90-Day Observation Feasibility

Use `merge_at` from the provider PR object as a UTC instant. Define `observation_start = merge_at`; qualifying evidence requires `observed_at > start and observed_at <= end`, where `end = merge_at + 90 × 24 hours`. Evidence stamped exactly at merge is excluded; evidence exactly at the end is included. A sample is mature only when the fixed audit cutoff is at or after `end` and the provider/git evidence scan completed. Timezone text is normalized to UTC; a missing/ambiguous merge timestamp is UNKNOWN.

If a repository disappears or history becomes inaccessible during the window, the evidence scan is incomplete: UNKNOWN, never LOW_OBSERVED. The same applies to inaccessible PR/event histories. Preserve the raw timestamp and provider timestamp rather than reinterpreting local timezone. AIDev's November-2025 cutoff means its timestamps alone are not enough to observe later 90-day outcomes; the acquisition snapshot must set and record a later audit cutoff.

## 7. Multi-Language Candidate Population

The live catalog supplies the only exact population total currently available: 2,743,854 PRs across 327,477 repositories. It also reports overall merged fractions by agent (80.5% aggregate), but it does not publish the requested language × repository-network × merged × size cross-tab. Therefore no exact language or 50-network conclusion is defensible yet.

| Population | Count now | Why |
|---|---:|---|
| Candidate | 2,743,854 catalog PRs (all source) | Published v4 total; enriched candidate scope is smaller. |
| Outcome-eligible | Unknown | Requires successful provider resolution, merge SHA/history, and full evidence scan. |
| Fully analyzable | Python only, count unknown | All relevant changed source languages need validated adapters and successful extraction. |
| Labelable | Unknown | Requires maturity and cached evidence; AIDev has no outcome table. |
| Final binary eligible | Unknown | Requires mature, reconstructable, fully analyzable HIGH or LOW_OBSERVED. |

The next phase must stream only needed columns from PR/repository tables, then join enriched file details for sampled/reachable candidates. It must report all five populations separately by language, repository/network, date, PR-size and changed-file strata. It must not use primary repository language as a substitute for changed-language eligibility.

## 8. Language Distribution

Natural distribution is not measured in this audit. The source describes the corpus as six-agent PRs and states that TypeScript is the most common language across agents, while Codex is more Python-skewed and Copilot more C#-skewed; that is contextual, not a training-population count. [AIDev replication package](https://github.com/SAILResearch/AI_Teammates_in_SE3)

Consequently, do not claim Python, JavaScript, or TypeScript dominance in the eligible cohort; do not balance. Report concentration by source repository ID, organization, frozen source-network ID, agent, changed language, and outcome availability after the scan. Python is the sole supported language today; every other target family remains catalog-only/unsupported until its adapter and parity gate exist.

## 9. Repository Distribution

The global PR/repository ratio (~8.4) does not show the tail concentration that matters for training. The AIDev-pop subset is intentionally selected from repositories with more than 100 stars, so it is additionally selection-biased. The 50-eligible-network target per language is plausible only after streaming counts and provider-reachability tests; it is **unverified for every language** in this audit. Apply `min(25 PRs, 2% of selected rows)` to the frozen network ID, report each exception, and keep all entities in a duplicate/fork/corrective group on one side of every split.

## 10. Attrition Model

```text
AIDev candidates
  → merged / PR identity resolvable
  → t0 base/head and merge history reconstructable
  → mature at frozen cutoff
  → repository and required historical objects accessible
  → changed languages all supported + adapter succeeds
  → complete post-merge evidence cache
  → HIGH or LOW_OBSERVED binary eligible
```

Counts are intentionally unknown after the first node. A preflight pilot should measure each transition and stratify failure reasons before any corpus-scale commitment. This is the correct result of an audit: estimates would fabricate observability that the source has not supplied.

## 11. Leakage Analysis

| Data source / field | Classification | Rule |
|---|---|---|
| Base/head trees, `base..head` diff, pre-t0 commits | Allowed for feature extraction at t0 | Pin to final PR head; never traverse post-merge commits. |
| AIDev pre-merge PR/file/commit metadata | Allowed for feature extraction at t0 | Use only data demonstrably available no later than t0; missing data remains missing. |
| Merge SHA/time, post-merge git history, revert/corrective PRs, timelines, incidents | Allowed only for label construction | Keep in label/evidence store, never in `features_42`. |
| Network/fork metadata, acquisition errors, source checksums, language/size/date strata | Allowed only for audit/stratification | Never expose as a model feature unless an independently t0-safe schema amendment is approved. |
| Post-t0 reviews, CI/check results, branch state, corrective patch/scope, label confidence, maturity/window completion | Forbidden | Do not select, impute, or calculate feature values from them. |

## 12. Acquisition Architecture Recommendation

Implement next, and only next, a small, restartable **preflight acquisition manifest** (not the final dataset):

```text
pinned AIDev catalog
  → repository/PR resolver
  → immutable PR candidate manifest
  → Git object + t0 snapshot/history acquisition
  → immutable outcome-evidence cache
  → evidence linker
  → policy labeler
  → fully-analyzable candidate report
```

Use provider node IDs, full URLs, SHA-256 checksums, ETags/response headers, retrieval time, API version, and content-addressed raw response/git-object locations. Queue repository work by frozen network, use bare shared object stores, cache negative results, and make every failure terminal and reportable. API use requires authenticated, backoff-aware batching: GitHub permits 5,000 REST requests/hour for a typical authenticated user but only 60/hour unauthenticated, and also imposes secondary limits. [GitHub rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)

At corpus scale this is necessarily asynchronous/batched and needs an evidence cache. Storage cannot be estimated credibly from AIDev alone because repository history size and retained evidence vary hugely; measure object-store bytes/PR, API calls/PR, clone failures, and wall time in the preflight. Create an immutable local snapshot/manifest before labeling; do not repeatedly query a moving provider state.

## 13. Risks and Failure Modes

1. AIDev has no verified adverse-outcome table; HIGH and LOW both depend on external, retained evidence.
2. Revert metadata is not universally structured. Manual reversals may be unobservable or ambiguous, biasing labels toward visible conventional reverts.
3. Repositories/objects may disappear or be rewritten after AIDev collection; failed history access must create UNKNOWN/exclusion, not LOW.
4. AIDev-pop and agent-authored PR selection are not a general GitHub sample.
5. Forks, mirrors, repeated patches, and corrective chains can leak across splits unless enriched before selection.
6. Only Python is fully analyzable now; multi-language claims would be invalid.
7. Provider rate limits, pagination, permissions, and evolving API behavior make raw caching, version capture, resumability, and failure accounting research requirements, not implementation details.
8. Merge strategy, force pushes, missing large patches, submodules, and LFS make the AIDev patch insufficient as a universal source snapshot.

## 14. Exact Next Step

Implement a **bounded, read-only acquisition preflight** for a fixed, stratified candidate sample (for example, 100–200 frozen PR identities), not a dataset builder. It must: pin/download only the required AIDev metadata columns and parquet schema/footer manifests; resolve PR/repository/fork-source/base/head and merge identifiers; attempt full-history retrieval; cache raw responses and failure reasons; scan each complete 90-day window; and emit the five-population attrition/concentration report. Review that report before admitting a language, creating labels at scale, or extracting any feature vector.

## Audit Completion Record

* **Files created:** `docs/phase_7f_data_acquisition_feasibility_audit.md`.
* **Files changed:** none.
* **Phase 7B files changed:** no.
* **42-feature schema changed:** no.
* **Labels/outcomes fabricated:** no; none were generated.
* **Commands/checks run:** local source/policy inspection; live AIDev card and pinned-file-tree inspection; GitHub REST documentation inspection. No tests were run because this audit changed documentation only.
* **Limitations:** no full AIDev parquet download/streaming scan, GitHub token, repository clone, or live provider acquisition was used. Exact language, network, size, outcome-eligible, labelable, and final-binary counts therefore remain explicitly unknown.

