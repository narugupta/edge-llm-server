# Step 2 results: rf022 vs ts15

This folder holds two batches of Step 2 (combined-load stress test) results
for Scenarios 1-3 across slots {1, 2, 4, 8}. They differ only in how
`continuous_stress.py` derived each workload type's arrival rate — the
scheduling policy under test (proxy.py) is identical between them.

## `rf022/` — Method A: equal utilization split

Run with `--rate-frac 0.22`. Every workload type's rate is derived as the
same fraction (22%) of its own isolated capacity from Step 1
(`rate = rate_frac * capacity / mean_service_time`).

**Limitation:** Batch's mean service time is 30-80x longer than Chat's, so
"equal utilization" translates into wildly unequal *arrival counts* —
Batch often got only 0-2 completed samples per run, too few to draw any
conclusion about Batch's SLO behaviour under contention.

## `ts15/` — Method B: equal completions (the corrected methodology)

Run with `--target-samples 15`. Rates are derived to aim for ~15
completed requests per type, scaling chat/moderate/batch down **together**
(never singling one out) only if that combined load would exceed a
sustainable utilization budget (`--combined-budget`, default 0.7).

This is the methodology used for the reported findings (e.g. the slots=1
total-starvation result) — sample counts came out balanced (9-28 per type
per run) instead of near-zero for Batch.

## Why both are kept

The two methods' results disagree on the Smart Preemption vs Fixed
Reservation Chat-latency ranking at slots=2 and slots=4 (see the email
sent 2026-08-27) — this disagreement is itself evidence that the ranking
isn't stable yet, not noise to average away. Keeping both batches lets
that comparison be re-checked later without re-running anything.

## Which one to trust for a new question

Default to `ts15/` — it's the corrected methodology with balanced sample
counts. Fall back to `rf022/` only when specifically checking whether a
finding is method-dependent (as in the disagreement above).