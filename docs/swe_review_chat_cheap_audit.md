# SWE-Review-Chat cheap audit

The experiment is isolated under `data/processed/swe_review_chat/`. The published schema (SWE-Review-Chat, train, 1,082,529 rows, 207 projects) contains `repo_full_name`, `pr_number`, `title`, `review_conversations`, `repo_url`, `pr_url`, `created_at`, `created_by`, `created_by_type`, `state`, `closed_at`, `merged_at`, `merged_by`, `merged_by_type`, `additions`, and `deletions`. `review_conversations` is a time-ordered list with `type`, `timestamp`, `reviewer`, `reviewer_type`, `body`, and review verdicts/inline discussion fields.

The labeler emits only `HIGHER_RISK=1` for a human `CHANGES_REQUESTED` verdict, and `LOWER_RISK=0` only for a human approval plus merged PR with no adverse human verdict. Bot/agent-only, conflicting, missing, malformed, and otherwise insufficient evidence is `UNKNOWN` and never becomes low risk. Review content, verdicts, outcomes, timestamps, reviewer identity, CI data, and provenance are label-only; `assert_no_leakage` rejects these from model inputs. The canonical 42-feature schema is imported but not modified.

The cohort selector is deterministic (seeded SHA-256 ordering), deduplicated, repository-diverse, and label-first. It is sized for a 6,000+ candidate pool and leaves natural-distribution holdout/grouped repository splitting to the later feature cohort. `DurablePRCache` provides per-PR atomic writes and checkpoints for resume-safe pilot/production execution; clone reuse, locks, bounded git operations, and pilot gates belong in the later runner.

The real audit streams the Hugging Face Parquet manifest with bounded batches; it does not load all 1.08M nested rows into RAM and does not clone repositories:

```text
python scripts/audit_swe_review_chat.py
```

The completed audit scanned 1,082,529 rows, found 1,082,389 unique PR identities across 207 repositories, and produced 67,151 HIGHER_RISK, 707,594 LOWER_RISK, and 307,784 UNKNOWN rows. There were 52,910 PRs with both a human CHANGES_REQUESTED and a later human APPROVED event; these remain HIGHER_RISK because the original proposal required corrective intervention. The maximum defensible candidate pool is 774,745 genuine binary-labeled rows, so a 3,000–5,000-row final supervised cohort is realistic before language filtering.

Reviewer types and event/verdict vocabularies are recorded in `reviewer_summary.json`. GitHub primary-language enrichment is separate, cached, and metadata-only; run `python scripts/enrich_swe_review_chat_languages.py` with `GITHUB_TOKEN` when needed.

With a local export, use `python scripts/audit_swe_review_chat.py --input PATH`. The exact later 50-PR pilot command is intentionally not executable yet: `python scripts/run_swe_review_chat.py --mode pilot50 --input data/processed/swe_review_chat/cohort.jsonl`.

The existing AIDev disposable checkout/cache under `data/processed/aidev_expansion/c/` was not touched. It contains cloned repository working trees, git metadata/objects, and lock files; it is safe to delete later only after confirming no AIDev process/checkpoint references it and retaining any required provenance. It is not safe to delete as part of this audit.
