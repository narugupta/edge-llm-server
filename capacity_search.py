#!/usr/bin/env python3
"""
capacity_search.py — Isolated theoretical capacity measurement (Step 1 of the
Aspect-6 finalization methodology)
=========================================================================
Finds the maximum number of *concurrent* requests of a single workload type
(Chat, Moderate, or Batch — never mixed) that the system can serve while
that type's own SLO still holds. This is the "theoretical maximum capacity" -
a system-level ceiling, independent of any scheduling policy, since with
only one type active there's nothing for a scheduler to arbitrate between.

Repeats every concurrency level multiple times (default 3) rather than
relying on a single wave, per the explicit warning in the meeting notes
that a single wave is not a reliable measurement.

Reuses run_one_client / PROMPTS / PRIORITY_LABEL from stress_test.py so the
request mechanics (prompts, headers, streaming TTFT measurement) are
identical to every other test already run in this project.

Bypasses the proxy by default (--port 8081, straight to llama-server) since
isolated single-type capacity is a property of the underlying server, not
of any particular scheduling policy. Point it at 8082 explicitly if you
specifically want to measure through the proxy instead.

Two search modes
  adaptive (default -- used whenever --concurrencies is NOT given)
    Exponential doubling from --start-n (1, 2, 4, 8, ...) until a
    concurrency level fails its SLO, then binary search between the last
    passing level and the first failing one to pin down the exact
    boundary. Converges to a precise answer without you having to guess
    a good --concurrencies range up front. If the SLO never breaks by
    --max-n, search stops there and reports capped_at_max_n=true.

  fixed grid (used whenever --concurrencies IS given)
    Tests exactly the comma-separated list you give it, nothing more,
    nothing adaptive. Useful for spot-checking specific values, or a
    manual second pass around a boundary an earlier adaptive run found.

Both modes share the same per-level measurement (run_concurrency_level):
--repeats trials pooled together, gated by BOTH the pooled mean AND every
individual trial passing -- one bad trial can't be averaged away by good
ones. Both modes also share the same non-monotonicity check via
find_max_concurrency(), applied identically regardless of which search
strategy chose the tested concurrency levels.

Usage
  # Terminal 1 — llama-server (must be running first)
  ./build/bin/llama-server \\
      -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \\
      --port 8081 --parallel 2 --ctx-size 2048 --kv-unified

  # Terminal 2 — capacity search (direct to server, bypassing the proxy)
  python3 capacity_search.py --client-type chat                      # adaptive (default)
  python3 capacity_search.py --client-type batch --start-n 10 --max-n 300
  python3 capacity_search.py --client-type moderate --concurrencies 10,20,30,40   # fixed grid

Repeat once per --slots value you want data for (restart llama-server with
a different --parallel between runs) to build the slot-count sweep the
meeting notes describe (1, 2, 4, 8 in the example).

Output JSON structure
  meta       — settings this run used, including search_mode
  levels     — one entry per concurrency level actually tested (in
                 adaptive mode this is the doubling + binary-search
                 trace, not every integer):
                 concurrency, repeats, per-trial summaries, pooled summary
                 across all repeats, slo_met (bool), raw per-request data
  derived    — max_concurrency_within_slo, capped_at_max_n (true if the
               SLO never broke within --max-n in adaptive mode; always
               false in fixed-grid mode), and a non_monotonic_warning if
               a smaller tested concurrency failed while a larger one
               passed (noise, worth a closer look rather than silently
               trusting)

SLO definitions (match the project's established targets exactly)
  chat      TTFT mean < 0.5s
  moderate  TTFT mean < 2.0s
  batch     total_time mean < 60.0s

An SLO is only counted as met if ALL requests at that concurrency
completed (zero errors) AND the mean latency of those completions is
under threshold. A concurrency level where half the requests time out but
the survivors are fast does NOT count as meeting the SLO -- this mirrors
the "violation rate is misleading under high load" lesson learned in the
earlier scale-sweep testing (failed requests must not be able to make a
result look better by dropping out of the average).
"""

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime

try:
    from stress_test import run_one_client, health_check
except ImportError:
    sys.exit(
        "Could not import from stress_test.py -- this script must be run "
        "from the same directory as stress_test.py (it reuses its client "
        "request/TTFT-measurement logic directly rather than duplicating it)."
    )

SLO_CHECK = {
    "chat":     {"metric": "ttft",       "threshold": 0.5,  "label": "TTFT mean < 0.5s"},
    "moderate": {"metric": "ttft",       "threshold": 2.0,  "label": "TTFT mean < 2.0s"},
    "batch":    {"metric": "total_time", "threshold": 60.0, "label": "Total mean < 60.0s"},
}


def run_trial(client_type: str, concurrency: int, trial_idx: int, host: str, port: int) -> list:
    """Fire `concurrency` simultaneous same-type requests, return raw results."""
    barrier = threading.Barrier(concurrency)
    results: list = []
    results_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=run_one_client,
            args=(client_type, i, trial_idx, host, port, barrier, results, results_lock),
        )
        for i in range(concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def evaluate_slo(client_type: str, results: list) -> dict:
    metric = SLO_CHECK[client_type]["metric"]
    threshold = SLO_CHECK[client_type]["threshold"]
    n_total = len(results)
    n_errors = sum(1 for r in results if r["error"])
    ok = [r for r in results if not r["error"] and r.get(metric) is not None]

    if ok:
        values = sorted(r[metric] for r in ok)
        mean_v = statistics.mean(values)
        median_v = statistics.median(values)
        p99_v = values[min(len(values) - 1, int(0.99 * len(values)))]
        max_v = values[-1]
    else:
        mean_v = median_v = p99_v = max_v = None

    slo_met = (n_errors == 0) and (mean_v is not None) and (mean_v < threshold)

    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "n_ok": len(ok),
        "completion_rate": round(len(ok) / n_total, 4) if n_total else None,
        "mean": mean_v,
        "median": median_v,
        "p99": p99_v,
        "max": max_v,
        "threshold": threshold,
        "slo_met": slo_met,
    }


def run_concurrency_level(client_type, concurrency, repeats, host, port, wave_delay, warmup=1) -> dict:
    # Warm-up: discard `warmup` throwaway trials before the real measurement,
    # so a cold prompt cache on the very first request of a run doesn't drag
    # a genuinely-passing level into a false failure. This mirrors the
    # project's own established convention (steady-state vs. first-round
    # numbers) rather than introducing a new one.
    for w in range(warmup):
        print(f"    warm-up {w + 1}/{warmup} (discarded) …")
        run_trial(client_type, concurrency, trial_idx=-1 - w, host=host, port=port)
        if wave_delay > 0:
            time.sleep(wave_delay)

    trial_summaries = []
    all_raw = []
    for r in range(repeats):
        raw = run_trial(client_type, concurrency, trial_idx=r, host=host, port=port)
        all_raw.extend(raw)
        summary = evaluate_slo(client_type, raw)
        trial_summaries.append(summary)
        mean_str = f"{summary['mean']:.3f}s" if summary["mean"] is not None else "n/a"
        status = "ok" if summary["slo_met"] else "FAIL"
        print(f"    trial {r + 1}/{repeats}: n_ok={summary['n_ok']:3d}/{summary['n_total']:3d}  "
              f"mean={mean_str:>8s}  [{status}]")
        if r < repeats - 1 and wave_delay > 0:
            time.sleep(wave_delay)

    # Pooled decision across all repeats (more reliable than any one trial,
    # per the "single wave is not reliable" warning) -- but ALSO require
    # every individual trial to have passed, so one bad trial can't be
    # averaged away by good ones.
    pooled_summary = evaluate_slo(client_type, all_raw)
    all_trials_met = all(t["slo_met"] for t in trial_summaries)

    return {
        "concurrency": concurrency,
        "repeats": repeats,
        "trials": trial_summaries,
        "pooled": pooled_summary,
        "slo_met": bool(pooled_summary["slo_met"] and all_trials_met),
        "raw": all_raw,
    }


def adaptive_search(client_type, host, port, repeats, wave_delay, start_n, max_n, warmup=1) -> tuple:
    """
    Exponential doubling from start_n (1, 2, 4, 8, ...) until a concurrency
    level fails its SLO, then binary search between the last passing level
    and the first failing one to pin down the exact boundary.

    Returns (levels_tested, capped_at_max_n) -- levels_tested is a list in
    the same shape run_concurrency_level() produces, so the existing
    find_max_concurrency() can be reused unchanged as the final boundary
    decision regardless of how the levels were chosen. This deliberately
    does NOT duplicate the pass/fail aggregation or monotonicity-check
    logic -- only the *search strategy* (which concurrency to test next)
    is new here.
    """
    tested: dict = {}  # concurrency -> level dict, avoids re-testing a value

    def test_level(n):
        if n in tested:
            return tested[n]
        print(f"\n  Concurrency {n} ({client_type}) — {repeats} repeated trials  [adaptive]")
        lvl = run_concurrency_level(client_type, n, repeats, host, port, wave_delay, warmup)
        tested[n] = lvl
        mean_str = f"{lvl['pooled']['mean']:.3f}s" if lvl['pooled']['mean'] is not None else "n/a"
        status = "PASS" if lvl["slo_met"] else "FAIL"
        print(f"    -> pooled: n_ok={lvl['pooled']['n_ok']}/{lvl['pooled']['n_total']}  "
              f"completion={lvl['pooled']['completion_rate']}  mean={mean_str}  [{status}]")
        time.sleep(wave_delay)
        return lvl

    # Phase 1: exponential doubling to bracket the boundary
    n = start_n
    last_pass, first_fail, capped = None, None, False
    while True:
        lvl = test_level(n)
        if lvl["slo_met"]:
            last_pass = n
            if n >= max_n:
                capped = True
                break
            next_n = min(n * 2, max_n)
            if next_n == n:
                capped = True
                break
            n = next_n
        else:
            first_fail = n
            break

    # Phase 2: binary search within the bracket, if we have one
    if last_pass is not None and first_fail is not None and not capped:
        lo, hi = last_pass, first_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lvl = test_level(mid)
            if lvl["slo_met"]:
                lo = mid
            else:
                hi = mid

    return list(tested.values()), capped


def find_max_concurrency(levels: list) -> dict:
    """
    Largest tested concurrency such that it AND every smaller tested
    concurrency also met the SLO. Flags non-monotonic results explicitly
    rather than silently trusting a passing level that skips over a
    failure at a smaller concurrency (measurement noise, worth a second
    look, not something to paper over).
    """
    levels_sorted = sorted(levels, key=lambda l: l["concurrency"])
    max_ok = None
    seen_failure = False
    non_monotonic = False
    for lvl in levels_sorted:
        if lvl["slo_met"]:
            if seen_failure:
                non_monotonic = True
            max_ok = lvl["concurrency"]
        else:
            seen_failure = True
    return {
        "max_concurrency_within_slo": max_ok,
        "non_monotonic_warning": non_monotonic,
        "note": (
            "non_monotonic_warning=true means a smaller tested concurrency "
            "failed its SLO while a larger one passed -- likely measurement "
            "noise (cold cache, thermal, background load). Worth rerunning "
            "the affected levels with more repeats before trusting "
            "max_concurrency_within_slo at face value."
            if non_monotonic else
            "Results were monotonic: every tested concurrency at or below "
            "max_concurrency_within_slo passed its SLO, every one above it "
            "(if any were tested) failed."
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Isolated per-type capacity search (Step 1 of the Aspect-6 methodology).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--client-type", required=True, choices=["chat", "moderate", "batch"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8081,
                    help="Default 8081 (direct to llama-server, bypassing the proxy) "
                         "since isolated capacity is a system property independent "
                         "of scheduling policy. Use 8082 to measure through the "
                         "proxy instead if that's specifically what you want.")
    p.add_argument("--slots", type=int, default=None,
                    help="Informational only -- records which --parallel value the "
                         "server was started with for this run's metadata. Does NOT "
                         "control the server; restart llama-server yourself with "
                         "the matching --parallel before running this script.")
    p.add_argument("--concurrencies", type=str, default=None,
                    help="Comma-separated concurrency levels, e.g. '5,10,20,30'. "
                         "If given, uses fixed-grid mode: tests exactly this list, "
                         "nothing more. If omitted, uses adaptive mode (exponential "
                         "doubling from --start-n, then binary search) instead.")
    p.add_argument("--start-n", type=int, default=1,
                    help="Adaptive mode only: starting concurrency for exponential "
                         "doubling (default 1). Note: Batch trials cost real "
                         "generation time (~25-50s) regardless of concurrency, so "
                         "starting at 1 for Batch wastes several minutes on levels "
                         "almost certain to pass -- consider --start-n 10 or higher "
                         "for Batch once you have any sense of where its ceiling is.")
    p.add_argument("--max-n", type=int, default=500,
                    help="Adaptive mode only: hard ceiling on concurrency to test "
                         "(default 500). If the SLO never breaks by this point, "
                         "search stops here and reports capped_at_max_n=true -- the "
                         "true ceiling is unknown, only that it's >= this value.")
    p.add_argument("--repeats", type=int, default=3,
                    help="Repeated trials per concurrency level (default 3). A "
                         "single wave is not a reliable measurement.")
    p.add_argument("--wave-delay", type=float, default=2.0,
                    help="Seconds to let the server settle between trials and "
                         "between concurrency levels (default 2.0).")
    p.add_argument("--warmup", type=int, default=1,
                    help="Throwaway trials to run and discard before the real "
                         "measurement at EVERY concurrency level (default 1), "
                         "so a cold prompt cache doesn't drag a genuinely-"
                         "passing level into a false failure -- matches this "
                         "project's established steady-state-vs-first-round "
                         "convention. Set to 0 to disable (e.g. if you "
                         "specifically want to include cold-start effects).")
    p.add_argument("--label", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label or f"capacity_{args.client_type}"
    output_path = args.output or f"results_{label}.json"
    search_mode = "fixed_grid" if args.concurrencies else "adaptive"

    print(f"\nedge-llm-server — Isolated Capacity Search ({label})")
    print(f"  Client type    : {args.client_type}  ({SLO_CHECK[args.client_type]['label']})")
    print(f"  Host:port      : {args.host}:{args.port}"
          f"{'  (direct to server)' if args.port != 8082 else '  (via proxy)'}")
    print(f"  Search mode    : {search_mode}")
    if search_mode == "fixed_grid":
        concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]
        print(f"  Concurrencies  : {concurrencies}")
    else:
        print(f"  Start / Max n  : {args.start_n} / {args.max_n}")
    print(f"  Repeats/level  : {args.repeats}  (+ {args.warmup} warm-up trial(s) discarded per level)")
    print(f"  Slots (info)   : {args.slots if args.slots is not None else 'not specified'}")
    print(f"  Started        : {datetime.now().isoformat(timespec='seconds')}")
    print("\n  Checking health …")
    if not health_check(args.host, args.port):
        sys.exit("  Server not reachable -- aborting.")

    capped_at_max_n = False
    if search_mode == "fixed_grid":
        levels = []
        for c in concurrencies:
            print(f"\n  Concurrency {c} ({args.client_type}) — {args.repeats} repeated trials")
            lvl = run_concurrency_level(args.client_type, c, args.repeats, args.host, args.port, args.wave_delay, args.warmup)
            levels.append(lvl)
            mean_str = f"{lvl['pooled']['mean']:.3f}s" if lvl['pooled']['mean'] is not None else "n/a"
            status = "PASS" if lvl["slo_met"] else "FAIL"
            print(f"    -> pooled: n_ok={lvl['pooled']['n_ok']}/{lvl['pooled']['n_total']}  "
                  f"completion={lvl['pooled']['completion_rate']}  mean={mean_str}  [{status}]")
            time.sleep(args.wave_delay)
    else:
        levels, capped_at_max_n = adaptive_search(
            args.client_type, args.host, args.port, args.repeats,
            args.wave_delay, args.start_n, args.max_n, args.warmup,
        )

    derived = find_max_concurrency(levels)
    derived["capped_at_max_n"] = capped_at_max_n
    if capped_at_max_n:
        derived["note"] += (
            f" Search stopped at --max-n={args.max_n} without the SLO ever "
            f"breaking -- the true ceiling is unknown, only that it is >= "
            f"{args.max_n}. Rerun with a higher --max-n if you need the "
            f"actual boundary."
        )

    print(f"\n{'=' * 74}")
    print(f"  CAPACITY SEARCH SUMMARY — {label}")
    print(f"{'=' * 74}")
    print(f"  Max concurrency within SLO: {derived['max_concurrency_within_slo']}"
          f"{'  (capped -- true ceiling may be higher)' if capped_at_max_n else ''}")
    if derived["non_monotonic_warning"]:
        print(f"  WARNING: non-monotonic results -- see 'note' field in output JSON")
    for lvl in sorted(levels, key=lambda l: l["concurrency"]):
        status = "PASS" if lvl["slo_met"] else "FAIL"
        mean_str = f"{lvl['pooled']['mean']:.3f}s" if lvl['pooled']['mean'] is not None else "n/a"
        print(f"    concurrency={lvl['concurrency']:4d}  mean={mean_str:>8s}  "
              f"completion={lvl['pooled']['completion_rate']}  [{status}]")

    output = {
        "meta": {
            "client_type": args.client_type,
            "host": args.host,
            "port": args.port,
            "via_proxy": args.port == 8082,
            "slots": args.slots,
            "search_mode": search_mode,
            "concurrencies": concurrencies if search_mode == "fixed_grid" else None,
            "start_n": args.start_n if search_mode == "adaptive" else None,
            "max_n": args.max_n if search_mode == "adaptive" else None,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "wave_delay": args.wave_delay,
            "slo_definition": SLO_CHECK[args.client_type],
            "label": label,
            "timestamp": datetime.now().isoformat(),
        },
        "levels": levels,
        "derived": derived,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {output_path}\n")


if __name__ == "__main__":
    main()