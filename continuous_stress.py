#!/usr/bin/env python3
"""
continuous_stress.py — Continuous mixed-workload stress test (Step 2)
======================================================================

Step 2 evaluates Chat, Moderate, and Batch under continuous mixed load
through the proxy. Unlike stress_test.py, requests are generated
continuously using independent Poisson arrival processes rather than
discrete waves.

Step 1 measured the capacity of each workload in isolation. Step 2 uses
those measurements to evaluate the scheduling policies under mixed load.

Scenarios currently supported by proxy.py:

    Scenario 1 — Priority Proxy
        --reserved-chat-slots 0

    Scenario 2 — Fixed Reservation
        --reserved-chat-slots N
        N > 0, without --smart-preemption

    Scenario 3 — Smart Preemption
        --reserved-chat-slots N --smart-preemption

        The current implementation does not interrupt requests that are
        already running. When Chat is waiting, the scheduler can reclaim
        a slot when a running request finishes and give it to Chat.

Scenarios 4 and 5 are not currently implemented in proxy.py:

    Scenario 4 — EDF (Earliest Deadline First)
    Scenario 5 — WFQ + Preemption

The test does not start or configure llama-server or proxy.py. Start
both separately with the required configuration before running this
script.

Server configuration:
    --parallel should match the --slots value used for the test.

Proxy configuration:
    Set --reserved-chat-slots and --smart-preemption according to the
    scenario being evaluated.

Example:

    Terminal 1 — llama-server

    ./build/bin/llama-server \
        -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \
        --port 8081 \
        --parallel 2 \
        --ctx-size 2048 \
        --kv-unified

    Terminal 2 — proxy

    python3 proxy.py --reserved-chat-slots 1

    Terminal 3 — continuous mixed load

    python3 continuous_stress.py \
        --scenario 2 \
        --slots 2 \
        --duration 90 \
        --chat-capacity-json results_capacity_chat_slots2.json \
        --moderate-capacity-json results_capacity_moderate_slots2.json \
        --batch-capacity-json results_capacity_batch_slots2.json \
        --rate-frac 0.5 \
        --reserved-chat-slots 1 \
        --output results_scenario2_slots2.json


Arrival rates
-------------

Arrival rates can be provided directly:

    --chat-rate
    --moderate-rate
    --batch-rate

They can also be derived from Step 1 capacity results using Little's
Law:

    rate = capacity / mean_service_time_at_capacity

The derived rate is multiplied by --rate-frac.

For each workload type, an explicitly supplied rate takes precedence
over the corresponding capacity JSON file.

If neither an explicit rate nor a capacity JSON file is provided for a
workload type, no requests of that type are generated.


Step 2 test goals
-----------------

The continuous mixed-load tests are used to evaluate:

1. The effect of fixed slot reservation on Moderate and Batch.

2. Scaling behaviour at different slot counts, particularly in light
   of the non-linear scaling observed during Step 1.

3. Scheduler behaviour when requests arrive at different times.

4. SLO compliance under sustained mixed workload.

5. Whether lower-priority workloads continue to make progress under
   higher-priority load.


Output
------

The output JSON contains:

    meta
        Test configuration, scenario, slot count, duration, arrival
        pattern, resolved arrival rates, and rate source.

    per_type
        Results for Chat, Moderate, and Batch:
        - n_arrived
        - n_completed
        - n_errors
        - completion_rate
        - mean SLO metric
        - p99 SLO metric
        - SLO violation rate
        - front-half mean
        - back-half mean

    raw
        Individual request results from stress_test.py:
        - client_id
        - client_type
        - priority
        - t_start
        - wave_arrival_s
        - ttft
        - total_time
        - tokens_out
        - tokens_per_s
        - error

Note:
    There are no discrete waves in this test. The field
    'wave_arrival_s' is retained for compatibility with the existing
    stress_test.py result format and represents the time since the
    corresponding workload generator started.
"""

import argparse
import json
import random
import statistics
import sys
import threading
import time
from datetime import datetime

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:
    sys.exit("requests not installed — run:  pip install requests")

try:
    from stress_test import run_one_client, health_check, PROMPTS, PRIORITY_LABEL, SLO_CHECK
except ImportError:
    sys.exit(
        "Could not import from stress_test.py -- this script must be run "
        "from the same directory as stress_test.py (it reuses its client "
        "request/TTFT-measurement logic and SLO definitions directly "
        "rather than duplicating them, the same way capacity_search.py does)."
    )

SCENARIO_LABELS = {
    1: "Priority Proxy",
    2: "Fixed Reservation",
    3: "Smart Preemption (v1 reactive, non-interrupting)",
    4: "EDF (not yet implemented)",
    5: "WFQ + Preemption (not yet implemented)",
}


# ── Rate resolution ──────────────────────────────────────────────────────────

def _capacity_and_mean_service_time(path: str) -> tuple:
    """Shared by the derive path and the explicit-rate sanity check: reads a
    capacity_search.py result JSON and returns (capacity, mean_service_time).

    IMPORTANT: "service time" here must be true slot-occupancy time --
    total_time -- not whatever SLO metric capacity_search.py happened to be
    checking at that concurrency. For chat/moderate, capacity_search.py's
    pooled.mean is TTFT (their SLO metric), which is far shorter than the
    actual time a request holds a slot -- using it would understate service
    time and overstate any derived rate or sustainability check by roughly
    ttft/total_time (often ~2x or more). So this always recomputes mean
    total_time directly from that level's raw per-request records, which
    capacity_search.py stores regardless of which metric was being checked."""
    with open(path) as f:
        d = json.load(f)
    capacity = d["derived"]["max_concurrency_within_slo"]
    if capacity is None or capacity <= 0:
        raise ValueError(f"{path}: no usable capacity (max_concurrency_within_slo="
                          f"{capacity!r}) to derive a rate from")
    level = next((l for l in d["levels"] if l["concurrency"] == capacity), None)
    if level is None:
        raise ValueError(f"{path}: capacity={capacity} but no matching level in "
                          f"'levels' to read service time from")
    total_times = [r["total_time"] for r in level.get("raw", [])
                   if r.get("total_time") is not None and not r.get("error")]
    if not total_times:
        raise ValueError(f"{path}: no usable total_time values in level {capacity}'s "
                          f"raw records to compute mean service time from")
    mean_service_time = statistics.mean(total_times)
    return capacity, mean_service_time


def derive_rate_from_capacity_json(path: str, rate_frac: float, slots: int) -> tuple:
    """Little's Law: rate = effective_servers / mean_service_time_at_capacity,
    scaled by rate_frac.

    FIX: effective_servers is min(capacity, slots), NOT capacity alone.
    `capacity` (max_concurrency_within_slo) is how many concurrent requests
    capacity_search.py found the system could absorb while still meeting the
    SLO metric -- it can legitimately exceed the physical slot count (e.g. a
    burst that queues briefly but still clears its SLO window). Treating
    capacity itself as the server count in Little's Law overstates sustained
    throughput: you cannot actually run more than `slots` requests
    concurrently for a *sustained* duration, no matter how much SLO
    headroom a short burst has. Capping at the true physical slot count is
    what makes the derived rate something continuous_stress.py's steady
    Poisson arrivals over `--duration` seconds can actually sustain.

    Returns (rate, detail_str, mean_service_time) -- mean_service_time is
    returned too so the caller can run the same utilization check used for
    explicit rates, without re-reading the file."""
    capacity, mean_service_time = _capacity_and_mean_service_time(path)
    effective_servers = min(capacity, slots)
    base_rate = effective_servers / mean_service_time
    rate = base_rate * rate_frac
    detail = (f"derived from {path}: capacity={capacity}, slots={slots}, "
              f"effective_servers=min(capacity,slots)={effective_servers}, "
              f"mean_total_time={mean_service_time:.3f}s (from raw records, "
              f"NOT pooled.mean which may be ttft), "
              f"base_rate={base_rate:.3f}/s, rate_frac={rate_frac} -> {rate:.3f}/s")
    return rate, detail, mean_service_time


def warn_if_unsustainable(client_type, rate, mean_service_time, slots, source, log):
    """Catches the sanity-test class of problem for ANY rate -- explicit or
    derived -- not just ones that went through Little's Law.

    Utilization = (rate * mean_service_time) / slots = the number of
    physical slots this rate would need, on average, to keep up, divided by
    the slots actually available. >1.0 means the rate is unsustainable in
    isolation: queue length and latency grow without bound over a long
    enough run, independent of scheduling policy (--reserved-chat-slots,
    --smart-preemption) and independent of the other two client types'
    load -- this is a floor, not a worst case."""
    if slots is None or slots <= 0 or mean_service_time is None:
        return
    needed_servers = rate * mean_service_time
    utilization = needed_servers / slots
    if utilization > 1.0:
        log(f"  WARNING: {client_type} rate {rate:.3f}/s needs ~{needed_servers:.2f} "
            f"concurrent slot(s) on average (rate x mean_service_time "
            f"{mean_service_time:.3f}s) but only {slots} physical slot(s) exist "
            f"(utilization={utilization:.2f}x, {source}). This rate is "
            f"UNSUSTAINABLE IN ISOLATION -- queue length and latency will grow "
            f"without bound the longer --duration runs, regardless of scheduling "
            f"policy or the other client types' load.")


def resolve_rate(client_type: str, explicit_rate, capacity_json, slots, rate_frac, log) -> tuple:
    """Returns (rate, detail, mean_service_time_or_None). mean_service_time is
    None only when no capacity_json was given at all (explicit rate, no
    sustainability data available) -- callers that need it for a combined
    cross-type utilization check should treat None as "unknown, can't check"."""
    if explicit_rate is not None:
        log(f"  {client_type:<9s} rate = {explicit_rate:.3f}/s (explicit)")
        if capacity_json:
            try:
                _, mean_service_time = _capacity_and_mean_service_time(capacity_json)
                warn_if_unsustainable(
                    client_type, explicit_rate, mean_service_time, slots,
                    source=f"explicit rate checked against {capacity_json}", log=log)
                return explicit_rate, "explicit", mean_service_time
            except (OSError, KeyError, ValueError) as exc:
                log(f"  {client_type:<9s} (could not sanity-check explicit rate "
                    f"against {capacity_json}: {exc})")
        return explicit_rate, "explicit", None
    if capacity_json:
        rate, detail, mean_service_time = derive_rate_from_capacity_json(
            capacity_json, rate_frac, slots)
        log(f"  {client_type:<9s} rate = {rate:.3f}/s ({detail})")
        warn_if_unsustainable(
            client_type, rate, mean_service_time, slots,
            source=f"derived from {capacity_json}", log=log)
        return rate, detail, mean_service_time
    log(f"  {client_type:<9s} rate = 0/s (no --{client_type}-rate or "
        f"--{client_type}-capacity-json given -- this type sends no load)")
    return 0.0, "none (no load)", None


def warn_if_combined_unsustainable(rates_and_mst: dict, slots, log):
    """Catches EXACTLY the bug the smart_preemption sanity test and the full
    S1-S3 x slots{1,2,4,8} matrix both hit: each type's rate can look
    individually sustainable (utilization <= 1.0 against `slots` on its
    own) while the combined demand still overloads the system, because
    chat/moderate/batch all draw from the SAME physical slot pool
    simultaneously rather than each having it to themselves.

    Per-type utilization must SUM to <=1.0 across all active types, not
    just individually. rates_and_mst: {client_type: (rate, mean_service_time)}
    -- types with mean_service_time=None (explicit rate, no capacity_json)
    are skipped in the sum with a note, since their true slot-time demand
    is unknown."""
    if slots is None or slots <= 0:
        return
    total_util = 0.0
    parts = []
    skipped = []
    for ct, (rate, mst) in rates_and_mst.items():
        if rate <= 0:
            continue
        if mst is None:
            skipped.append(ct)
            continue
        u = (rate * mst) / slots
        total_util += u
        parts.append(f"{ct}={u:.3f}")
    if skipped:
        log(f"  (combined-utilization check: skipping {', '.join(skipped)} -- "
            f"explicit rate with no capacity_json, true service time unknown)")
    if total_util > 1.0:
        log(f"  WARNING: COMBINED utilization across all active types = "
            f"{total_util:.3f}x ({', '.join(parts)}) on {slots} shared slot(s). "
            f"Each type's rate may look sustainable ALONE, but chat/moderate/"
            f"batch all draw from the SAME {slots} physical slot(s) "
            f"simultaneously -- their utilizations must SUM to <=1.0, not "
            f"each individually. This combined load is UNSUSTAINABLE: queues "
            f"will grow without bound across all three types as --duration "
            f"runs longer, regardless of scheduling policy. Lower --rate-frac "
            f"(or split it across types) until this sum is comfortably <1.0.")
    elif parts:
        log(f"  Combined utilization across active types = {total_util:.3f}x "
            f"({', '.join(parts)}) on {slots} shared slot(s) -- sustainable.")


def resolve_rates_target_samples(target_samples, duration, slots, capacity_jsons,
                                  explicit_rates, budget, log):
    """Alternative to the equal-utilization split in resolve_rate(): aims for
    ~target_samples completed requests PER TYPE over `duration`, then scales
    ALL types down together (preserving their ratio, i.e. still roughly equal
    sample counts, just fewer) only if the naive equal-count rates would
    exceed `budget` combined utilization.

    Why this exists: equal utilization (old approach) means
    utilization = rate * mean_service_time is the same for every type, but
    since batch's mean_service_time is 30-80x chat's, equal utilization
    forces batch's rate (and therefore its sample count) to be 30-80x
    smaller than chat's -- a 45s run can get 30+ chat samples and 0 batch
    samples. Equal SAMPLE COUNT is what you actually need for a fair
    scenario comparison, so this derives rate = target_samples / duration
    for every type first, then checks the combined utilization that implies
    and scales everything down together (never just one type) if it's over
    budget -- so if scaling is needed, every type falls equally short of
    target_samples rather than batch being singled out again.

    explicit_rates: {client_type: rate_or_None} -- types with an explicit
    rate are left untouched; their utilization (if a matching capacity_json
    is available) is subtracted from `budget` before splitting the rest.

    Returns {client_type: (rate, detail, mean_service_time_or_None)}."""
    types = ("chat", "moderate", "batch")
    mst = {}
    naive_rate = {}
    naive_util = {}
    fixed_util = 0.0

    for ct in types:
        if explicit_rates.get(ct) is not None:
            cj = capacity_jsons.get(ct)
            if cj:
                try:
                    _, m = _capacity_and_mean_service_time(cj)
                    fixed_util += (explicit_rates[ct] * m) / slots
                except (OSError, KeyError, ValueError):
                    pass
            continue
        cj = capacity_jsons.get(ct)
        if not cj:
            continue
        try:
            _, m = _capacity_and_mean_service_time(cj)
        except (OSError, KeyError, ValueError) as exc:
            log(f"  {ct:<9s} (could not compute target-samples rate: {exc})")
            continue
        mst[ct] = m
        naive_rate[ct] = target_samples / duration
        naive_util[ct] = (naive_rate[ct] * m) / slots

    total_naive = sum(naive_util.values())
    remaining_budget = max(0.0, budget - fixed_util)
    scale = 1.0
    if total_naive > 0 and total_naive > remaining_budget:
        scale = remaining_budget / total_naive

    results = {}
    for ct in types:
        if explicit_rates.get(ct) is not None:
            _, m = (None, None)
            cj = capacity_jsons.get(ct)
            if cj:
                try:
                    _, m = _capacity_and_mean_service_time(cj)
                except (OSError, KeyError, ValueError):
                    pass
            log(f"  {ct:<9s} rate = {explicit_rates[ct]:.3f}/s (explicit, "
                f"target-samples mode leaves it untouched)")
            results[ct] = (explicit_rates[ct], "explicit", m)
            continue
        if ct not in naive_rate:
            log(f"  {ct:<9s} rate = 0/s (no --{ct}-rate or --{ct}-capacity-json "
                f"given -- this type sends no load)")
            results[ct] = (0.0, "none (no load)", None)
            continue
        r = naive_rate[ct] * scale
        expected_n = r * duration
        detail = (f"target-samples: aiming for {target_samples} over {duration:.0f}s "
                  f"(naive_rate={naive_rate[ct]:.4f}/s), scaled {scale:.3f}x for "
                  f"combined budget {budget} -> rate={r:.4f}/s, expected_n~={expected_n:.1f}")
        log(f"  {ct:<9s} rate = {r:.4f}/s ({detail})")
        results[ct] = (r, detail, mst[ct])

    if scale < 1.0:
        log(f"  NOTE: naive equal-{target_samples}-sample rates implied "
            f"{total_naive:.3f}x combined utilization, over the "
            f"{remaining_budget:.3f}x budget remaining after explicit rates -- "
            f"ALL target-samples types scaled down together by {scale:.3f}x. "
            f"Expected samples per type will be below {target_samples}; "
            f"increase --duration to recover the full target without raising "
            f"utilization.")
    return results


# ── Arrival generator ─────────────────────────────────────────────────────────

def arrival_generator(client_type, rate, duration, pattern, session, host, port,
                       timeout, results, results_lock, in_flight_counter, log):
    """Fire requests for `client_type` for `duration` seconds at `rate` req/s,
    fire-and-forget (each request runs in its own thread so the generator
    never blocks waiting for a response). Returns the list of spawned
    threads so the caller can join them during drain.

    Uses stress_test.py's run_one_client with barrier=None -- the exact
    calling convention it already supports for fire-immediately, caller-
    controlled timing, confirmed compatible before this refactor."""
    threads = []
    if rate <= 0:
        return threads

    t0 = time.perf_counter()
    n_fired = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= duration:
            break
        if pattern == "poisson":
            gap = random.expovariate(rate)
        else:  # fixed
            gap = 1.0 / rate
        time.sleep(min(gap, max(0.0, duration - elapsed)))
        elapsed = time.perf_counter() - t0
        if elapsed >= duration:
            break

        with in_flight_counter["lock"]:
            in_flight_counter[client_type] += 1
        t = threading.Thread(
            target=_send_and_decrement,
            args=(client_type, n_fired, host, port, results, results_lock,
                  timeout, session, t0, in_flight_counter),
            daemon=True,
        )
        threads.append(t)
        t.start()
        n_fired += 1

    log(f"  [{client_type}] generator done: fired {n_fired} arrivals over "
        f"{duration}s (target rate {rate:.3f}/s)")
    return threads


def _send_and_decrement(client_type, instance_id, host, port, results,
                         results_lock, timeout, session, t0, in_flight_counter):
    try:
        run_one_client(client_type, instance_id, wave_idx=0, host=host, port=port,
                        barrier=None, results=results, results_lock=results_lock,
                        wave_t0=t0, timeout=timeout, session=session)
    finally:
        with in_flight_counter["lock"]:
            in_flight_counter[client_type] -= 1


# ── Evaluation ────────────────────────────────────────────────────────────────

def summarise(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return {"n": 0, "mean": None, "p99": None, "max": None}
    sv = sorted(valid)
    p99_idx = max(0, int(len(sv) * 0.99) - 1)
    return {"n": len(valid), "mean": round(statistics.mean(valid), 4),
            "p99": round(sv[p99_idx], 4), "max": round(max(valid), 4)}


def evaluate_type(client_type, results):
    metric = SLO_CHECK[client_type]["metric"]
    threshold = SLO_CHECK[client_type]["threshold"]
    typed = [r for r in results if r["client_type"] == client_type]
    typed_sorted = sorted(typed, key=lambda r: r["wave_arrival_s"])
    n_arrived = len(typed_sorted)
    errors = [r for r in typed_sorted if r["error"]]
    ok = [r for r in typed_sorted if not r["error"] and r.get(metric) is not None]
    values = [r[metric] for r in ok]
    stats = summarise(values)

    violations = sum(1 for v in values if v > threshold) + len(errors)
    violation_rate = round(violations / n_arrived, 4) if n_arrived else None

    half = len(typed_sorted) // 2
    front = [r[metric] for r in typed_sorted[:half] if not r["error"] and r.get(metric) is not None]
    back = [r[metric] for r in typed_sorted[half:] if not r["error"] and r.get(metric) is not None]

    return {
        "metric": metric,
        "threshold": threshold,
        "n_arrived": n_arrived,
        "n_completed": len(ok),
        "n_errors": len(errors),
        "error_samples": sorted({e["error"] for e in errors})[:5],
        "completion_rate": round(len(ok) / n_arrived, 4) if n_arrived else None,
        "stats": stats,
        "slo_violation_rate": violation_rate,
        "front_half_mean": round(statistics.mean(front), 4) if front else None,
        "back_half_mean": round(statistics.mean(back), 4) if back else None,
        "degrading_over_run": (
            front and back and statistics.mean(back) > statistics.mean(front) * 1.2
        ),
    }


# ── CLI / main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed-workload continuous-arrival stress test (Step 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help="Informational label only -- does not configure the proxy. "
                         "See module docstring for the Scenario -> proxy.py flag mapping.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8082,
                    help="Default 8082 -- the PROXY, not the raw server (unlike "
                         "capacity_search.py). Scenario comparisons are only "
                         "meaningful through the scheduling layer.")
    p.add_argument("--slots", type=int, required=True,
                    help="Informational only -- must match what you started "
                         "llama-server with via --parallel.")
    p.add_argument("--reserved-chat-slots", type=int, default=None,
                    help="Informational only -- must match what you started proxy.py "
                         "with. 0 = Scenario 1 (Priority Proxy), >0 = Scenario 2 "
                         "(Fixed Reservation) UNLESS --smart-preemption is also set.")
    p.add_argument("--smart-preemption", action="store_true",
                    help="Informational only -- set this if proxy.py was started "
                         "WITH --smart-preemption (Scenario 3). Recorded in metadata "
                         "so a run can be correctly attributed to Scenario 2 vs 3 "
                         "later, since both use --reserved-chat-slots.")
    p.add_argument("--duration", type=float, default=90.0,
                    help="Seconds of continuous arrivals per type (default 90).")
    p.add_argument("--arrival-pattern", default="poisson", choices=["poisson", "fixed"])
    p.add_argument("--drain-timeout", type=float, default=180.0,
                    help="Max seconds to wait for in-flight requests to finish "
                         "after arrivals stop (default 180).")
    p.add_argument("--timeout", type=float, default=150.0, help="Per-request HTTP timeout.")

    for ct in ("chat", "moderate", "batch"):
        p.add_argument(f"--{ct}-rate", type=float, default=None,
                        help=f"{ct} arrivals/second, explicit. Overrides --{ct}-capacity-json.")
        p.add_argument(f"--{ct}-capacity-json", type=str, default=None,
                        help=f"Path to a capacity_search.py result JSON for {ct}; "
                             f"rate derived via Little's Law if --{ct}-rate not given.")
    p.add_argument("--rate-frac", type=float, default=1.0,
                    help="Multiplier applied to every rate derived from a "
                         "--*-capacity-json (default 1.0 = 100%% of implied "
                         "capacity throughput). Has no effect on explicit --*-rate. "
                         "IGNORED if --target-samples is set (use --combined-budget "
                         "instead in that mode).")
    p.add_argument("--target-samples", type=int, default=None,
                    help="If set, switches rate derivation to aim for roughly this "
                         "many completed samples PER TYPE over --duration, instead of "
                         "an equal utilization split. Equal utilization (--rate-frac "
                         "applied uniformly) gives wildly unequal sample counts when "
                         "service times differ by orders of magnitude (e.g. batch's "
                         "~50s vs chat's ~1s) -- a 45s run can get 0 batch samples "
                         "while chat gets 30+. This mode fixes that by targeting equal "
                         "counts, then scaling ALL types down together (preserving "
                         "their ratio) only if that would exceed --combined-budget. "
                         "Types with an explicit --*-rate are left alone and their "
                         "utilization (if a matching --*-capacity-json is given) is "
                         "reserved out of the budget before splitting the rest.")
    p.add_argument("--combined-budget", type=float, default=0.7,
                    help="Target combined utilization ceiling (summed across all "
                         "active types, see warn_if_combined_unsustainable) used only "
                         "in --target-samples mode. Default 0.7.")

    p.add_argument("--label", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def health_check(host, port) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        print(f"  Reachable — status: {r.json().get('status', '')!r}")
        return True
    except Exception as exc:
        print(f"  Health check failed: {exc}")
        return False


def main():
    args = parse_args()
    label = args.label or f"scenario{args.scenario}_slots{args.slots}"
    output_path = args.output or f"results_{label}.json"

    def log(msg):
        print(msg)

    print(f"\nedge-llm-server — Continuous Mixed-Workload Stress Test (Step 2)")
    print(f"  Label          : {label}")
    print(f"  Scenario       : {args.scenario} ({SCENARIO_LABELS[args.scenario]})")
    print(f"  Target         : {args.host}:{args.port} "
          f"{'(proxy)' if args.port != 8081 else '(WARNING: looks like the raw server, not the proxy)'}")
    print(f"  Slots (info)   : {args.slots}")
    print(f"  Reservation    : reserved_slots={args.reserved_chat_slots}  "
          f"smart_preemption={args.smart_preemption}  (informational -- must match "
          f"what proxy.py was actually started with)")
    print(f"  Duration       : {args.duration}s per type, {args.arrival_pattern} arrivals")
    print(f"  Started        : {datetime.now().isoformat(timespec='seconds')}\n")

    print("  Resolving arrival rates …")
    rates, rate_details = {}, {}
    rates_and_mst = {}
    if args.target_samples is not None:
        print(f"  (target-samples mode: aiming for {args.target_samples} samples/type, "
              f"combined-budget={args.combined_budget})")
        capacity_jsons = {ct: getattr(args, f"{ct}_capacity_json") for ct in ("chat", "moderate", "batch")}
        explicit_rates = {ct: getattr(args, f"{ct}_rate") for ct in ("chat", "moderate", "batch")}
        resolved = resolve_rates_target_samples(
            args.target_samples, args.duration, args.slots, capacity_jsons,
            explicit_rates, args.combined_budget, log)
        for ct, (rate, detail, mst) in resolved.items():
            rates[ct] = rate
            rate_details[ct] = detail
            rates_and_mst[ct] = (rate, mst)
    else:
        for ct in ("chat", "moderate", "batch"):
            explicit = getattr(args, f"{ct}_rate")
            cap_json = getattr(args, f"{ct}_capacity_json")
            rate, detail, mst = resolve_rate(ct, explicit, cap_json, args.slots, args.rate_frac, log)
            rates[ct] = rate
            rate_details[ct] = detail
            rates_and_mst[ct] = (rate, mst)

    warn_if_combined_unsustainable(rates_and_mst, args.slots, log)

    print("\n  Checking health …")
    if not health_check(args.host, args.port):
        sys.exit("  Proxy/server not reachable -- aborting.")

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=256, pool_maxsize=256)
    session.mount("http://", adapter)

    results = []
    results_lock = threading.Lock()
    in_flight = {"chat": 0, "moderate": 0, "batch": 0, "lock": threading.Lock()}

    print(f"\n  Generating arrivals for {args.duration}s …")
    gen_threads = []
    all_request_threads = []
    for ct in ("chat", "moderate", "batch"):
        gt = threading.Thread(
            target=lambda ct=ct: all_request_threads.extend(arrival_generator(
                ct, rates[ct], args.duration, args.arrival_pattern, session,
                args.host, args.port, args.timeout, results, results_lock,
                in_flight, log,
            )),
            daemon=True,
        )
        gen_threads.append(gt)
        gt.start()
    for gt in gen_threads:
        gt.join()

    print(f"\n  Arrivals complete. Draining in-flight requests "
          f"(up to {args.drain_timeout}s) …")
    drain_deadline = time.perf_counter() + args.drain_timeout
    for t in all_request_threads:
        remaining = drain_deadline - time.perf_counter()
        if remaining <= 0:
            break
        t.join(timeout=remaining)
    still_running = sum(1 for t in all_request_threads if t.is_alive())
    if still_running:
        print(f"  WARNING: {still_running} request(s) still in flight after "
              f"drain timeout -- their results are incomplete/missing, not "
              f"silently counted as failures. Increase --drain-timeout if "
              f"this recurs.")

    per_type = {ct: evaluate_type(ct, results) for ct in ("chat", "moderate", "batch")}

    print(f"\n{'=' * 78}")
    print(f"  CONTINUOUS STRESS SUMMARY — {label}")
    print(f"{'=' * 78}")
    for ct in ("chat", "moderate", "batch"):
        s = per_type[ct]
        mean_str = f"{s['stats']['mean']:.3f}s" if s['stats']['mean'] is not None else "n/a"
        drift = " [DEGRADING OVER RUN]" if s["degrading_over_run"] else ""
        print(f"  {ct:<9s} arrived={s['n_arrived']:4d}  completed={s['n_completed']:4d}  "
              f"completion={s['completion_rate']}  {s['metric']} mean={mean_str:>8s}  "
              f"violation_rate={s['slo_violation_rate']}{drift}")

    output = {
        "meta": {
            "label": label,
            "scenario": args.scenario,
            "scenario_label": SCENARIO_LABELS[args.scenario],
            "host": args.host,
            "port": args.port,
            "slots": args.slots,
            "reserved_chat_slots": args.reserved_chat_slots,
            "smart_preemption": args.smart_preemption,
            "duration": args.duration,
            "arrival_pattern": args.arrival_pattern,
            "drain_timeout": args.drain_timeout,
            "timeout": args.timeout,
            "rates": rates,
            "rate_resolution": rate_details,
            "rate_frac": args.rate_frac,
            "still_in_flight_after_drain": still_running,
            "timestamp": datetime.now().isoformat(),
        },
        "per_type": per_type,
        "raw": sorted(results, key=lambda r: r["wave_arrival_s"]),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved -> {output_path}\n")


if __name__ == "__main__":
    main()