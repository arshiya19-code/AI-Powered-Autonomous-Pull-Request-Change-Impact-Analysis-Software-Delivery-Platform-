# Phase 7C dataset contract and candidate audit

Phase 7C defines contracts and an audit only. It does not acquire a large PR
population, build a training corpus, balance classes, train a model, or add a
language adapter.

## Linked tables

`samples` records the provider/repository/PR identity, network and fork
relationships, base/head/merge revisions, the analysis timestamp, and source
snapshot provenance. Its head revision is `t0`: the final PR head commit
immediately before merge. Patch identity, repeated commit group, and related
corrective/revert group preserve future duplicate and provenance enforcement.

`analysis_metadata` separately records all changed languages, primary changed
language, file and line shares, repository language, adapter versions/status,
coverage, unsupported languages, and extraction result. Detection is not
eligibility: a language is eligible only with a real adapter and a passed
quality gate. The current default matrix makes Python the only fully
analyzable language.

`features_42` is an isolated model-input table. Its exact ordered allowlist is
derived from `dataset.config.ALL_FEATURES`; it admits no metadata, labels, or
post-`t0` outcome fields. `labels` keeps outcome state, confidence, evidence,
observation bounds, and policy version. `quality` records inclusion, explicit
exclusion, parse/analysis failures, duplicate/fork clusters, and missingness.
`manifest` records versions, source revision, deterministic seed (42), quotas,
network caps, exclusions, windows/cutoff, support matrix, checksums, and
candidate/selected/attrition counts.

## Temporal and label policy

The default observation and maturity periods are both 90 days. A PR is
`merged_mature`, `merged_immature`, `unmerged`, `unavailable`, or
`conflicting`. Unmerged, immature, unavailable, and conflicting observations
never become `low_observed`.

Labels are deterministic: confirmed post-`t0` rollback evidence is `high`; a
materially overlapping corrective change is `medium`; only a mature merged PR
with sufficient no-adverse evidence is `low_observed`; incomplete evidence is
`unknown`; materially contradictory evidence is `conflicted`.

## Candidate audit and selection policy

The audit reports availability and attrition, natural and eligible language
distribution, repository/network/fork distribution, candidate-derived size
strata, and label prevalence before any future balancing decision. It reports
prevalence by language, repository stratum, PR-size stratum, and period.

The planning default is 50 eligible repository networks per language. Per
network selection is capped at `min(25, 2% of selected rows)` and shortages
are reported rather than fabricated. The audit has no sampling mutation,
oversampling, undersampling, synthesis, relabeling, or training functionality.

## Phase 7D: outcome evidence and labels

AIDev v4 is the candidate-population source, not the outcome-label source. A
separate cache-first `OutcomeEvidenceProvider` collects raw, post-merge records
in the fixed 90-day interval after `merged_at`. Each record carries its source,
retrieval time, reference, classification, and explicit linkage method/reason.
Network collection is optional; an unavailable provider produces `unknown`,
never `low_observed`. The current implementation ships only a deterministic
local-cache provider and an explicit unavailable provider—there is no mutable
live-GitHub assumption.

The reconciler gives confirmed rollback/revert evidence precedence (`high`),
then materially overlapping, explicitly linked corrective evidence (`medium`).
Material conflicts remain `conflicted`. A mature PR can be `low_observed` only
after its evidence provider confirms the complete window was evaluated and
found no qualifying evidence. This is “no observed adverse outcome,” not proof
of safety.

## Phase 7E: candidate construction and splits

Selection is deterministic (seed 42), preserves natural labels, requires full
language coverage and successful analysis, and applies the stated per-network
cap `min(25, 2% of selected rows)`. Exact patch identity is used when present;
otherwise head SHA and then provider PR identity are explicit fallback groups.
Repository names remain provisional networks until fork-parent metadata is
reproducibly enriched and cached.

`phase7e_dataset_builder` creates linked candidate-table records and an
attrition report but deliberately leaves `features_42` empty: Phase 7E does not
perform uncontrolled extraction. External splits hold out whole networks;
temporal splits enforce an embargo and reject network, duplicate, and related
event leakage.

## Current limitation and next boundary

The repository contains no pinned AIDev v4 source snapshot or repository-history
evidence cache. Therefore this is **phase foundation complete—ready for
controlled data acquisition**, not a final dataset. A controlled next phase
must pin the AIDev revision, populate source checksums, acquire reproducible
history evidence, then invoke the existing Phase 7B extraction interfaces only
for selected, fully supported candidates.

For an offline, non-mutating source audit, run
`python scripts/audit_aidev_candidate_population.py INPUT.json REPORT.json --cutoff 2026-01-01T00:00:00Z --source-revision PINNED_REVISION`.
The report contains population, language, size, temporal, duplicate, provisional
network, label-availability, and attrition fields; it does not claim a final
dataset exists.

### Legacy-artifact warning

`data/processed/aidev_pilot_audit.json` documents a previous 25-row,
patch-only pilot with a `high_risk`/`low_risk` field. Its own audit shows zero
fully analysed PRs and missing graph-derived values; moreover, that legacy
field is not the Phase 7D post-merge adverse-outcome target. It is retained as
historical evidence only and is explicitly excluded from the Phase 7C/7D/7E
candidate and label path.
