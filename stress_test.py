#!/usr/bin/env python3
"""
stress_test.py — Priority-ordering stress test for edge-llm-server proxy
=========================================================================
Goal: verify that priority ordering holds under genuine backpressure
(more concurrent clients than server slots) and observe Batch throughput
degradation trend under sustained load.

Two arrival patterns are supported:

  simultaneous (default, original behaviour)
    All clients in a wave fire at the exact same instant via a
    threading.Barrier. This is the harshest test of the scheduler —
    it exposes microsecond-level dispatch races.

  staggered (new)
    Clients in a wave are launched one at a time, in randomised order,
    with a randomised delay between launches (default ~0.3s +/- 0.1s
    jitter). This is a secondary realism test: in real usage, a chat
    request doesn't usually arrive in the same millisecond as a batch
    job — it arrives while other things are already in flight. The
    question this pattern answers is different from simultaneous-fire:
    not "who wins a dead-heat race" but "does a higher-priority request
    correctly jump ahead of already-queued lower-priority work, and
    does Batch ever get starved indefinitely under continuous load."

IMPORTANT — violation semantics differ between the two patterns
  Under simultaneous arrival, low-priority TTFT < high-priority TTFT
  in the same wave is always a scheduler failure (both started the
  race together).

  Under staggered arrival, this is NOT always true. If Batch arrives
  first and both slots are idle, it correctly gets dispatched
  immediately — that is not a violation, it's correct use of idle
  capacity. A real violation under staggered arrival is: a
  higher-priority request was ALREADY WAITING when a lower-priority
  request got dispatched ahead of it.

  We don't have proxy-side dispatch timestamps on the client side (the
  proxy's /proxy/results records queue_wait per request but doesn't
  expose which client_id maps to which req_id back to this script), so
  we can't compute the exact causal ordering. Instead, for staggered
  runs we report each violation WITH its arrival gap (delta_arrival_s)
  so a human can judge whether it's explainable by arrival timing
  (large gap, idle-slot pickup — not a bug) or whether the arrival gap
  is small but the wrong client still won (genuine same-window failure
  — same mechanism as a simultaneous-arrival violation).

  As a rule of thumb: a violation where the lower-priority client
  arrived more than ~1s before the higher-priority client is very
  likely explainable by idle-slot pickup, not a scheduling failure.

Usage
  # Terminal 1 — llama-server (must be running first)
  ./build/bin/llama-server \\
      -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \\
      --port 8081 --parallel 2 --ctx-size 2048 --kv-unified

  # Terminal 2 — proxy (must be running, with arrival window tuned)
  python3 proxy.py --arrival-window 0.005 --dispatch-gap 0.005

  # Terminal 3 — stress test
  python3 stress_test.py                                   # simultaneous (default)
  python3 stress_test.py --arrival-pattern staggered        # staggered, defaults
  python3 stress_test.py --arrival-pattern staggered \\
      --stagger-delay 0.5 --stagger-jitter 0.2             # slower, more spread out
  python3 stress_test.py --waves 10 --clients 8             # heavier load
  python3 stress_test.py --port 8081                        # direct to server (no proxy)

Arguments
  --host            server/proxy host (default: localhost)
  --port            proxy port (default: 8082); use 8081 to bypass proxy
  --server-port     actual server port recorded in metadata (default: 8081)
  --waves           number of waves (default: 8)
  --clients         total clients per wave (default: 6)
  --chat-frac       fraction of clients that are chat/high-priority (default: 0.33)
  --batch-frac      fraction that are batch/low-priority (default: 0.34)
                    remainder are moderate/medium-priority
  --wave-delay      seconds between waves (default: 1.0)
  --arrival-pattern simultaneous | staggered (default: simultaneous)
  --stagger-delay   staggered mode: mean seconds between successive client
                    launches within a wave (default: 0.3)
  --stagger-jitter  staggered mode: +/- random jitter added to each delay,
                    seconds (default: 0.1)
  --output          output JSON path (default: results_stress_<label>.json)
  --label           run label (default: stress)

Client mix example (--clients 6, default fractions):
  chat     × 2  (high  priority)
  moderate × 2  (medium priority)
  batch    × 2  (low   priority)

Priority-ordering check
  Within each wave, for every (high, low) and (high, medium) and
  (medium, low) pair of clients, we check whether the higher-priority
  client got its first token before the lower-priority one.  A
  violation is recorded when the lower-priority client wins.  For
  staggered runs, each violation also records the arrival gap between
  the two clients (delta_arrival_s) — see note above on interpretation.

Batch starvation check (staggered mode only)
  Tracks the longest single wait observed for any batch client
  (arrival to first token).  If this grows wave-over-wave without
  bound under sustained staggered load, that's a sign Batch could be
  starved indefinitely.  Reported as a simple max/trend, not a formal
  proof — true starvation analysis needs proxy-side instrumentation
  (see module docstring above).
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
import threading
from datetime import datetime
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests not installed — run:  pip install requests")

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {
    "chat": "What is the capital of France? Answer in one sentence.",
    "moderate": (
        "Explain the difference between a process and a thread in an "
        "operating system. Cover scheduling, memory isolation, and "
        "communication overhead. Keep the answer under 150 words."
    ),
    "batch": (
        "Write a detailed technical explanation of how transformer-based "
        "language models work. Cover the attention mechanism, positional "
        "encoding, feed-forward layers, layer normalisation, and how "
        "autoregressive decoding produces tokens one at a time. Discuss "
        "why memory bandwidth rather than compute is often the bottleneck "
        "during token generation on edge devices with limited DRAM "
        "bandwidth. Include a brief comparison between full-precision "
        "FP16 inference and INT4 quantised inference in terms of model "
        "size, throughput, and output quality. Aim for roughly 400 words."
    ),
}

PRIORITY_LABEL = {"chat": "high", "moderate": "medium", "batch": "low"}
PRIORITY_INT   = {"chat": 0,      "moderate": 1,         "batch": 2}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stress test for priority proxy")
    p.add_argument("--host",        default="localhost")
    p.add_argument("--port",        default=8082, type=int)
    p.add_argument("--server-port", default=8081, type=int)
    p.add_argument("--waves",       default=8,    type=int,
                   help="Number of waves (default: 8)")
    p.add_argument("--clients",     default=6,    type=int,
                   help="Total clients per wave (default: 6)")
    p.add_argument("--chat-frac",   default=0.33, type=float,
                   help="Fraction of clients that are Chat (default: 0.33)")
    p.add_argument("--batch-frac",  default=0.34, type=float,
                   help="Fraction that are Batch (default: 0.34)")
    p.add_argument("--wave-delay",  default=1.0,  type=float,
                   help="Seconds between waves (default: 1.0)")
    p.add_argument("--arrival-pattern", default="simultaneous",
                   choices=["simultaneous", "staggered"],
                   help=(
                       "simultaneous: all clients in a wave fire at once "
                       "(original behaviour, harshest race-condition test). "
                       "staggered: clients launch one at a time in randomised "
                       "order with randomised delay (realism test). "
                       "(default: simultaneous)"
                   ))
    p.add_argument("--stagger-delay", default=0.3, type=float,
                   metavar="SECONDS",
                   help=(
                       "staggered mode only: mean seconds between successive "
                       "client launches within a wave. (default: 0.3)"
                   ))
    p.add_argument("--stagger-jitter", default=0.1, type=float,
                   metavar="SECONDS",
                   help=(
                       "staggered mode only: +/- random jitter applied to "
                       "each stagger delay, seconds. (default: 0.1)"
                   ))
    p.add_argument("--label",       default="stress")
    p.add_argument("--output",      default=None)
    return p.parse_args()


def build_client_mix(n_clients: int, chat_frac: float, batch_frac: float) -> list[str]:
    """
    Return a list of client types of length n_clients,
    e.g. ['chat', 'chat', 'moderate', 'moderate', 'batch', 'batch'].
    """
    n_chat     = max(1, round(n_clients * chat_frac))
    n_batch    = max(1, round(n_clients * batch_frac))
    n_moderate = max(0, n_clients - n_chat - n_batch)
    mix = (["chat"] * n_chat +
           ["moderate"] * n_moderate +
           ["batch"] * n_batch)
    # trim or pad to exact n_clients
    while len(mix) < n_clients:
        mix.append("moderate")
    return mix[:n_clients]


# ── Single client ─────────────────────────────────────────────────────────────

def run_one_client(
    client_type: str,
    instance_id: int,          # distinguishes e.g. chat-0, chat-1
    wave_idx: int,
    host: str,
    port: int,
    barrier: Optional[threading.Barrier],
    results: list,
    results_lock: threading.Lock,
    wave_t0: Optional[float] = None,
) -> None:
    """
    Fire one client's request and record timing.

    If `barrier` is given, the client waits on it before firing — this
    is simultaneous-arrival mode, where all clients in the wave are
    released at the same instant.

    If `barrier` is None, the client fires immediately when this
    function is called — the caller (run_wave_staggered) controls
    timing by staggering when each thread is started. `wave_t0`, if
    given, is used to record how long after the wave began this client
    actually arrived (for interpreting staggered-mode violations).
    """
    client_id = f"{client_type}-{instance_id}"
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": PROMPTS[client_type]}],
        "max_tokens": 512,
        "stream": True,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Priority":   PRIORITY_LABEL[client_type],
        "X-Client-Id":  client_type,
    }

    result = {
        "wave":             wave_idx,
        "client_id":        client_id,
        "client_type":      client_type,
        "priority":         PRIORITY_INT[client_type],
        "t_start":          None,
        "wave_arrival_s":   None,   # seconds after wave began that this client fired
        "ttft":             None,
        "total_time":       None,
        "tokens_out":       0,
        "tokens_per_s":     None,
        "error":            None,
    }

    if barrier is not None:
        barrier.wait()   # simultaneous mode — release with the rest of the wave

    t_start = time.perf_counter()
    result["t_start"] = t_start
    if wave_t0 is not None:
        result["wave_arrival_s"] = round(t_start - wave_t0, 4)

    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True, timeout=180
        ) as resp:
            resp.raise_for_status()
            first_token = True
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    if first_token:
                        result["ttft"] = time.perf_counter() - t_start
                        first_token = False
                    result["tokens_out"] += 1

        result["total_time"] = time.perf_counter() - t_start
        if result["total_time"] and result["tokens_out"] > 0:
            gen_time = result["total_time"] - (result["ttft"] or 0)
            if gen_time > 0:
                result["tokens_per_s"] = round(result["tokens_out"] / gen_time, 2)

    except requests.exceptions.ConnectionError:
        result["error"] = "ConnectionError"
    except requests.exceptions.Timeout:
        result["error"] = "Timeout (180s)"
    except requests.exceptions.HTTPError as exc:
        result["error"] = f"HTTP {exc.response.status_code}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    with results_lock:
        results.append(result)


# ── Wave runners ──────────────────────────────────────────────────────────────

def run_wave(
    wave_idx: int,
    client_mix: list[str],
    host: str,
    port: int,
) -> list[dict]:
    """Simultaneous-arrival wave: every client fires at the same instant."""
    results      = []
    results_lock = threading.Lock()
    # +1 so the main thread also waits at the barrier, keeping timing tight
    barrier = threading.Barrier(len(client_mix) + 1)

    type_count: dict[str, int] = {}
    threads = []
    for ctype in client_mix:
        instance_id = type_count.get(ctype, 0)
        type_count[ctype] = instance_id + 1
        t = threading.Thread(
            target=run_one_client,
            args=(ctype, instance_id, wave_idx, host, port,
                  barrier, results, results_lock, None),
            daemon=True,
        )
        threads.append(t)
        t.start()

    barrier.wait()   # release all clients at once
    for t in threads:
        t.join(timeout=200)

    return results


def run_wave_staggered(
    wave_idx: int,
    client_mix: list[str],
    host: str,
    port: int,
    stagger_delay: float,
    stagger_jitter: float,
) -> list[dict]:
    """
    Staggered-arrival wave: clients fire one at a time, in randomised
    order, with a randomised delay between launches. No barrier is
    used — each thread fires as soon as it is started.

    Order is shuffled per wave so that priority class does not predict
    arrival order, matching real usage where a chat request can arrive
    at any point relative to an in-flight batch job.
    """
    results      = []
    results_lock = threading.Lock()
    threads      = []

    shuffled = client_mix.copy()
    random.shuffle(shuffled)

    type_count: dict[str, int] = {}
    wave_t0 = time.perf_counter()

    for i, ctype in enumerate(shuffled):
        instance_id = type_count.get(ctype, 0)
        type_count[ctype] = instance_id + 1

        t = threading.Thread(
            target=run_one_client,
            args=(ctype, instance_id, wave_idx, host, port,
                  None, results, results_lock, wave_t0),
            daemon=True,
        )
        threads.append(t)
        t.start()

        if i < len(shuffled) - 1:
            delay = stagger_delay + random.uniform(-stagger_jitter, stagger_jitter)
            time.sleep(max(0.0, delay))

    for t in threads:
        t.join(timeout=200)

    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def summarise(values: list[Optional[float]]) -> dict:
    valid = [v for v in values if v is not None]
    if not valid:
        return {"n": 0, "mean": None, "median": None,
                "min": None, "max": None, "stdev": None, "p99": None}
    sv = sorted(valid)
    p99_idx = max(0, int(len(sv) * 0.99) - 1)
    return {
        "n":      len(valid),
        "mean":   round(statistics.mean(valid), 3),
        "median": round(statistics.median(valid), 3),
        "min":    round(min(valid), 3),
        "max":    round(max(valid), 3),
        "stdev":  round(statistics.stdev(valid), 3) if len(valid) > 1 else 0.0,
        "p99":    round(sv[p99_idx], 3),
    }


# ── Priority-ordering analysis ────────────────────────────────────────────────

def analyse_priority_ordering(all_results: list[dict], arrival_pattern: str) -> dict:
    """
    For each wave, check whether higher-priority clients got lower TTFT
    than lower-priority clients.

    For simultaneous-arrival runs, every violation is a genuine
    scheduler failure (both clients started the race together).

    For staggered-arrival runs, each violation also records the
    arrival gap between the two clients (delta_arrival_s — positive
    means the lower-priority client arrived first). A large gap
    (roughly >1s) usually means the lower-priority client correctly
    picked up an idle slot before the higher-priority client even
    arrived — not a scheduler failure. A small or negative gap with
    the lower-priority client still winning is the same failure mode
    as the simultaneous-arrival case. We report the gap rather than
    auto-classifying, since we don't have proxy-side dispatch
    timestamps on the client side to be certain.

    Returns:
      violations: list of dicts describing each inversion
      violation_rate: fraction of pairwise comparisons that were violated
      per_wave_violations: count per wave
      likely_idle_pickup / likely_genuine: only populated for staggered
        runs — a rough split based on the >1s arrival-gap heuristic
    """
    violations = []
    total_pairs = 0
    per_wave: dict[int, int] = {}

    waves: dict[int, list[dict]] = {}
    for r in all_results:
        waves.setdefault(r["wave"], []).append(r)

    priority_pairs = [("chat", "moderate"), ("chat", "batch"), ("moderate", "batch")]

    for wave_idx, wave_results in waves.items():
        wave_violations = 0
        for high_type, low_type in priority_pairs:
            high_clients = [r for r in wave_results
                            if r["client_type"] == high_type and r["ttft"] is not None]
            low_clients  = [r for r in wave_results
                            if r["client_type"] == low_type  and r["ttft"] is not None]
            for h in high_clients:
                for l in low_clients:
                    total_pairs += 1
                    if l["ttft"] < h["ttft"]:
                        wave_violations += 1
                        delta_arrival = None
                        if (h.get("wave_arrival_s") is not None and
                                l.get("wave_arrival_s") is not None):
                            # positive = low arrived first (possible idle pickup)
                            delta_arrival = round(
                                h["wave_arrival_s"] - l["wave_arrival_s"], 3
                            )
                        violations.append({
                            "wave":            wave_idx,
                            "high_client":     h["client_id"],
                            "high_ttft":       h["ttft"],
                            "high_arrival_s":  h.get("wave_arrival_s"),
                            "low_client":      l["client_id"],
                            "low_ttft":        l["ttft"],
                            "low_arrival_s":   l.get("wave_arrival_s"),
                            "delta_s":         round(h["ttft"] - l["ttft"], 4),
                            "delta_arrival_s": delta_arrival,
                            "pair":            f"{high_type}>{low_type}",
                        })
        per_wave[wave_idx] = wave_violations

    result = {
        "total_pairs":      total_pairs,
        "total_violations": len(violations),
        "violation_rate":   round(len(violations) / total_pairs, 4) if total_pairs else 0,
        "per_wave":         per_wave,
        "violations":       violations,
    }

    if arrival_pattern == "staggered":
        IDLE_PICKUP_THRESHOLD_S = 1.0
        likely_idle = [v for v in violations
                       if v["delta_arrival_s"] is not None
                       and v["delta_arrival_s"] > IDLE_PICKUP_THRESHOLD_S]
        likely_genuine = [v for v in violations
                          if v["delta_arrival_s"] is None
                          or v["delta_arrival_s"] <= IDLE_PICKUP_THRESHOLD_S]
        result["likely_idle_pickup_count"] = len(likely_idle)
        result["likely_genuine_count"]     = len(likely_genuine)
        result["idle_pickup_threshold_s"]  = IDLE_PICKUP_THRESHOLD_S
        result["note"] = (
            "Heuristic split based on arrival gap, not confirmed via proxy "
            "dispatch timestamps. 'likely_idle_pickup' = low-priority client "
            f"arrived >{IDLE_PICKUP_THRESHOLD_S}s before the high-priority "
            "client (probably picked up an idle slot correctly). "
            "'likely_genuine' = arrival gap was small or unknown — same "
            "failure mode as a simultaneous-arrival violation."
        )

    return result


# ── Batch starvation check (staggered mode) ───────────────────────────────────

def analyse_batch_starvation(all_results: list[dict]) -> dict:
    """
    Tracks batch client wait (arrival to first token) across waves.
    This is an upper bound on "starvation" — it includes queue_wait
    AND prompt-eval compute time, since we don't have a clean
    queue_wait-only signal on the client side.

    Reports the max and the per-wave trend. A flag is raised if the
    max TTFT in the back half of the run is markedly higher than the
    front half, which would suggest worsening starvation under
    sustained load rather than just normal queueing.
    """
    batch_rows = [r for r in all_results
                  if r["client_type"] == "batch" and r.get("ttft") is not None]
    if not batch_rows:
        return {"n": 0}

    by_wave: dict[int, list[float]] = {}
    for r in batch_rows:
        by_wave.setdefault(r["wave"], []).append(r["ttft"])

    per_wave_max = {w: round(max(v), 3) for w, v in sorted(by_wave.items())}
    all_ttfts    = [r["ttft"] for r in batch_rows]

    waves_sorted = sorted(per_wave_max.keys())
    half = max(1, len(waves_sorted) // 2)
    front_half = [per_wave_max[w] for w in waves_sorted[:half]]
    back_half  = [per_wave_max[w] for w in waves_sorted[half:]]

    front_mean = statistics.mean(front_half) if front_half else 0
    back_mean  = statistics.mean(back_half) if back_half else 0
    worsening  = back_mean > front_mean * 1.5  # arbitrary but explicit threshold

    return {
        "n":                  len(batch_rows),
        "max_ttft_overall":   round(max(all_ttfts), 3),
        "mean_ttft_overall":  round(statistics.mean(all_ttfts), 3),
        "per_wave_max_ttft":  per_wave_max,
        "front_half_mean":    round(front_mean, 3),
        "back_half_mean":     round(back_mean, 3),
        "worsening_flag":     worsening,
        "note": (
            "max_ttft is an upper bound on wait time (includes compute, not "
            "just queueing). worsening_flag fires if the back-half mean of "
            "per-wave max TTFT exceeds 1.5x the front-half mean — a rough "
            "signal only, not a formal starvation proof."
        ),
    }


# ── Throughput trend ──────────────────────────────────────────────────────────

def throughput_trend(all_results: list[dict]) -> dict:
    """
    Returns mean tokens/s for batch clients per wave, to reveal any
    degradation under sustained load (thermal or contention effect).
    """
    waves: dict[int, list] = {}
    for r in all_results:
        if r["client_type"] == "batch" and r.get("tokens_per_s") is not None:
            waves.setdefault(r["wave"], []).append(r["tokens_per_s"])
    per_wave = {w: round(statistics.mean(v), 2) for w, v in sorted(waves.items())}
    values = list(per_wave.values())
    trend = "degrading" if (len(values) >= 3 and values[-1] < values[0]) else "stable"
    return {"per_wave_mean_tps": per_wave, "trend": trend}


# ── Printing ──────────────────────────────────────────────────────────────────

COL = {
    "chat":     "\033[92m",
    "moderate": "\033[93m",
    "batch":    "\033[94m",
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "red":      "\033[91m",
    "green":    "\033[92m",
    "dim":      "\033[2m",
    "yellow":   "\033[93m",
}
R = COL["reset"]


def _c(ctype, text):
    return f"{COL.get(ctype, '')}{text}{R}"


def print_wave(wave_idx: int, results: list[dict], arrival_pattern: str) -> None:
    print(f"\n  {'─'*80}")
    print(f"  Wave {wave_idx + 1}")
    if arrival_pattern == "staggered":
        print(f"  {'Client':<16} {'Priority':<10} {'Arrival(s)':>11} {'TTFT (s)':>10} "
              f"{'Total (s)':>10} {'t/s':>8} {'Tokens':>7}")
    else:
        print(f"  {'Client':<16} {'Priority':<10} {'TTFT (s)':>10} "
              f"{'Total (s)':>10} {'t/s':>8} {'Tokens':>7}")
    print(f"  {'─'*80}")

    sort_key = (lambda x: x.get("wave_arrival_s") or 0) if arrival_pattern == "staggered" \
        else (lambda x: x["priority"])

    for r in sorted(results, key=sort_key):
        ctype = r["client_type"]
        if r["error"]:
            print(f"  {_c(ctype, r['client_id']):<25} "
                  f"{COL['red']}ERROR: {r['error']}{R}")
            continue
        ttft  = f"{r['ttft']:.3f}"       if r["ttft"]       else "—"
        total = f"{r['total_time']:.3f}" if r["total_time"] else "—"
        tps   = f"{r['tokens_per_s']:.1f}" if r["tokens_per_s"] else "—"
        if arrival_pattern == "staggered":
            arr = f"{r['wave_arrival_s']:.3f}" if r.get("wave_arrival_s") is not None else "—"
            print(f"  {_c(ctype, r['client_id']):<25}"
                  f" {PRIORITY_LABEL[ctype]:<10}"
                  f" {arr:>11} {ttft:>10} {total:>10} {tps:>8} {r['tokens_out']:>7}")
        else:
            print(f"  {_c(ctype, r['client_id']):<25}"
                  f" {PRIORITY_LABEL[ctype]:<10}"
                  f" {ttft:>10} {total:>10} {tps:>8} {r['tokens_out']:>7}")


def print_final_summary(
    all_results: list[dict],
    ordering: dict,
    throughput: dict,
    starvation: Optional[dict],
    label: str,
    client_mix: list[str],
    n_waves: int,
    arrival_pattern: str,
) -> None:
    bold = COL["bold"]
    print(f"\n{bold}{'═'*72}{R}")
    print(f"{bold}  STRESS TEST SUMMARY — {label} [{arrival_pattern}]{R}")
    print(f"  {n_waves} waves × {len(client_mix)} clients  "
          f"({client_mix.count('chat')} chat, "
          f"{client_mix.count('moderate')} moderate, "
          f"{client_mix.count('batch')} batch per wave)")
    print(f"{bold}{'═'*72}{R}\n")

    print(f"  {bold}Latency by priority class{R}")
    print(f"  {'Type':<12} {'TTFT mean':>10} {'TTFT p99':>10} "
          f"{'TTFT max':>10} {'Total mean':>12} {'mean t/s':>10}")
    print(f"  {'─'*72}")
    for ctype in ["chat", "moderate", "batch"]:
        rows = [r for r in all_results
                if r["client_type"] == ctype and r["error"] is None]
        if not rows:
            print(f"  {_c(ctype, ctype):<20}  no data")
            continue
        st  = summarise([r["ttft"]        for r in rows])
        tt  = summarise([r["total_time"]  for r in rows])
        tps = summarise([r["tokens_per_s"] for r in rows if r["tokens_per_s"]])
        print(f"  {_c(ctype, ctype):<20}"
              f" {st['mean']:>10.3f}"
              f" {st['p99']:>10.3f}"
              f" {st['max']:>10.3f}"
              f" {tt['mean']:>12.3f}"
              f" {(tps['mean'] if tps['mean'] else 0):>10.1f}")
    print()

    total_v = ordering["total_violations"]
    total_p = ordering["total_pairs"]
    rate    = ordering["violation_rate"] * 100
    color   = COL["red"] if total_v > 0 else COL["green"]
    print(f"  {bold}Priority-ordering violations{R}")
    print(f"  {color}{total_v} / {total_p} pairwise comparisons violated "
          f"({rate:.1f}%){R}")

    if arrival_pattern == "staggered" and total_v > 0:
        idle  = ordering.get("likely_idle_pickup_count", 0)
        genu  = ordering.get("likely_genuine_count", 0)
        thresh = ordering.get("idle_pickup_threshold_s", 1.0)
        print(f"  {COL['dim']}  of which: {idle} likely idle-slot pickup "
              f"(low-priority arrived >{thresh}s earlier — probably not a "
              f"bug), {genu} likely genuine (small/zero arrival gap){R}")

    if ordering["per_wave"]:
        per_wave_str = "  Wave violations: " + ", ".join(
            f"W{w+1}={v}" for w, v in sorted(ordering["per_wave"].items())
        )
        print(per_wave_str)
    if ordering["violations"]:
        print(f"\n  Top violations (by delta):")
        top = sorted(ordering["violations"],
                     key=lambda x: x["delta_s"], reverse=True)[:5]
        for v in top:
            arr_note = ""
            if arrival_pattern == "staggered" and v.get("delta_arrival_s") is not None:
                arr_note = f"  arrival_gap={v['delta_arrival_s']:.3f}s"
            print(f"    Wave {v['wave']+1}: {v['high_client']} TTFT={v['high_ttft']:.3f}s "
                  f"> {v['low_client']} TTFT={v['low_ttft']:.3f}s  "
                  f"(delta={v['delta_s']:.3f}s)  [{v['pair']}]{arr_note}")
    print()

    print(f"  {bold}Batch throughput trend (tokens/s per wave){R}")
    tps_by_wave = throughput["per_wave_mean_tps"]
    if tps_by_wave:
        for w, tps in sorted(tps_by_wave.items()):
            bar = "█" * int(tps / 2)
            print(f"    Wave {w+1:>2}: {tps:>6.1f} t/s  {bar}")
        trend_color = COL["yellow"] if throughput["trend"] == "degrading" else COL["green"]
        print(f"  Trend: {trend_color}{throughput['trend']}{R}")
        print(f"  {COL['dim']}Note: this label compares wave 1 to the last wave only — "
              f"check per-wave values above before drawing conclusions, "
              f"especially on a machine that was already warm at the start "
              f"of this run.{R}")
    else:
        print("    no batch data")
    print()

    if starvation is not None and starvation.get("n", 0) > 0:
        print(f"  {bold}Batch wait/starvation check (staggered mode){R}")
        print(f"    Max TTFT observed   : {starvation['max_ttft_overall']:.3f}s")
        print(f"    Mean TTFT overall   : {starvation['mean_ttft_overall']:.3f}s")
        print(f"    Front-half wave mean: {starvation['front_half_mean']:.3f}s")
        print(f"    Back-half wave mean : {starvation['back_half_mean']:.3f}s")
        flag_color = COL["red"] if starvation["worsening_flag"] else COL["green"]
        flag_text  = "WORSENING" if starvation["worsening_flag"] else "no worsening trend"
        print(f"    {flag_color}{flag_text}{R}")
        print(f"    {COL['dim']}{starvation['note']}{R}")
        print()


# ── Health check ──────────────────────────────────────────────────────────────

def health_check(host: str, port: int) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        status = r.json().get("status", "")
        print(f"  Reachable — status: {status!r}")
        return True
    except Exception as exc:
        print(f"  {COL['red']}Health check failed: {exc}{R}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_path = args.output or f"results_{args.label}.json"
    via_proxy   = args.port != args.server_port

    client_mix = build_client_mix(args.clients, args.chat_frac, args.batch_frac)

    bold = COL["bold"]
    print(f"\n{bold}edge-llm-server — Stress Test ({args.label}){R}")
    print(f"  Host           : {args.host}:{args.port}")
    print(f"  Via proxy      : {'yes' if via_proxy else 'no (direct to server)'}")
    print(f"  Arrival pattern: {args.arrival_pattern}")
    if args.arrival_pattern == "staggered":
        print(f"  Stagger delay  : {args.stagger_delay}s +/- {args.stagger_jitter}s")
    print(f"  Waves          : {args.waves}")
    print(f"  Clients/wave   : {args.clients}  "
          f"({client_mix.count('chat')} chat, "
          f"{client_mix.count('moderate')} moderate, "
          f"{client_mix.count('batch')} batch)")
    print(f"  Wave delay     : {args.wave_delay}s")
    print(f"  Started        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("  Checking health …")
    if not health_check(args.host, args.port):
        if via_proxy:
            print("\n  Start the proxy first:  python3 proxy.py --arrival-window 0.005")
        else:
            print("\n  Start the server first:")
            print("    ./build/bin/llama-server \\")
            print("      -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \\")
            print("      --port 8081 --parallel 2 --ctx-size 2048 --kv-unified")
        sys.exit(1)

    all_results: list[dict] = []

    for wave_idx in range(args.waves):
        print(f"\n  Firing wave {wave_idx + 1}/{args.waves} "
              f"[{args.arrival_pattern}] …  ({len(client_mix)} clients)")
        if args.arrival_pattern == "staggered":
            wave_results = run_wave_staggered(
                wave_idx, client_mix, args.host, args.port,
                args.stagger_delay, args.stagger_jitter,
            )
        else:
            wave_results = run_wave(wave_idx, client_mix, args.host, args.port)
        all_results.extend(wave_results)
        print_wave(wave_idx, wave_results, args.arrival_pattern)
        if wave_idx < args.waves - 1:
            time.sleep(args.wave_delay)

    ordering   = analyse_priority_ordering(all_results, args.arrival_pattern)
    throughput = throughput_trend(all_results)
    starvation = (analyse_batch_starvation(all_results)
                  if args.arrival_pattern == "staggered" else None)

    print_final_summary(
        all_results, ordering, throughput, starvation,
        args.label, client_mix, args.waves, args.arrival_pattern,
    )

    output = {
        "meta": {
            "host":            args.host,
            "port":            args.port,
            "server_port":     args.server_port,
            "waves":           args.waves,
            "clients":         args.clients,
            "client_mix":      client_mix,
            "wave_delay":      args.wave_delay,
            "arrival_pattern": args.arrival_pattern,
            "stagger_delay":   args.stagger_delay if args.arrival_pattern == "staggered" else None,
            "stagger_jitter":  args.stagger_jitter if args.arrival_pattern == "staggered" else None,
            "label":           args.label,
            "via_proxy":       via_proxy,
            "timestamp":       datetime.now().isoformat(),
        },
        "raw":               all_results,
        "priority_ordering": ordering,
        "throughput_trend":  throughput,
        "batch_starvation":  starvation,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved → {output_path}\n")


if __name__ == "__main__":
    main()