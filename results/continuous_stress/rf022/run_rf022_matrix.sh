#!/usr/bin/env bash
# run_rf022_matrix.sh — ARCHIVED, frozen snapshot for reproducibility only.
#
# This is the script that actually produced the rf022/ results, reconstructed
# from the assistant conversation on 2026-08-26/27 (verified against the real
# terminal output from that run: same RATE_FRAC=0.22, DURATION=45, output
# filenames step2_scenarioN_slotsS_rf022, and the same derived rates it
# printed at the time -- e.g. chat=0.310/s, moderate=0.045/s, batch=0.004/s
# at slots=1).
#
# It was later edited IN PLACE into what is now run_step2_s1_s3_matrix.sh
# (the --target-samples / ts15 methodology) once the equal-utilization
# split was found to starve Batch of samples -- see
# results/continuous_stress/README.md for why both methods are kept.
#
# DO NOT use this script for new runs -- it reproduces a methodology known
# to give unbalanced sample counts (see README). It exists purely so the
# rf022/ folder's provenance is traceable and re-runnable if ever needed.
#
# Requires: results_capacity_{chat,moderate,batch}_slots{1,2,4,8}.json
# present in the working directory (from capacity-search.zip).

set -uo pipefail  # NOT -e: one failed run should not kill the whole matrix

MODEL="${MODEL:-$HOME/Qwen2.5-1.5B-Q4_K_M.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./build/bin/llama-server}"
SERVER_PORT=8081
PROXY_PORT=8082
DURATION=45           # seconds of load per run
RATE_FRAC=0.22        # combined across chat+moderate+batch sharing the same
                      # slots must sum to <1.0 -- NOT per-type headroom.
                      # KNOWN LIMITATION (see README): applying this
                      # fraction equally to every type's OWN isolated
                      # capacity gives Batch a far lower absolute arrival
                      # rate than Chat, since Batch's service time is
                      # 30-80x longer -- Batch often got 0-2 samples/run.
DRAIN_TIMEOUT=180
WARMUP_SECONDS=5
OUTDIR="results/step2"

mkdir -p "$OUTDIR"

SLOT_COUNTS=(1 2 4 8)
SCENARIOS=(1 2 3)

server_pid=""
proxy_pid=""

cleanup() {
    [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

start_server() {
    local slots=$1
    echo "  [server] starting llama-server --parallel $slots ..."
    "$LLAMA_SERVER_BIN" -m "$MODEL" --port "$SERVER_PORT" \
        --parallel "$slots" --ctx-size 2048 --kv-unified \
        > "server_log_step2_slots${slots}.txt" 2>&1 &
    server_pid=$!
    sleep "$WARMUP_SECONDS"
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "  [server] FAILED to start (check server_log_step2_slots${slots}.txt) -- skipping slots=$slots"
        return 1
    fi
    return 0
}

stop_server() {
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
    server_pid=""
    sleep 1
}

start_proxy() {
    local scenario=$1 slots=$2
    case "$scenario" in
        1) proxy_args=(--reserved-chat-slots 0) ;;
        2) proxy_args=(--reserved-chat-slots 1) ;;
        3) proxy_args=(--smart-preemption) ;;
        *) echo "  [proxy] unknown scenario $scenario"; return 1 ;;
    esac
    echo "  [proxy] starting: python3 proxy.py --slots $slots ${proxy_args[*]}"
    python3 proxy.py --slots "$slots" "${proxy_args[@]}" \
        > "proxy_log_step2_s${scenario}_slots${slots}.txt" 2>&1 &
    proxy_pid=$!
    sleep 2
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
        echo "  [proxy] FAILED to start (check proxy_log_step2_s${scenario}_slots${slots}.txt)"
        return 1
    fi
    return 0
}

stop_proxy() {
    [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null
    wait "$proxy_pid" 2>/dev/null
    proxy_pid=""
    sleep 1
}

run_stress() {
    local scenario=$1 slots=$2
    local rf_tag
    rf_tag=$(echo "$RATE_FRAC" | tr -d '.')
    local label="step2_scenario${scenario}_slots${slots}_rf${rf_tag}"
    local output="${OUTDIR}/results_${label}.json"

    local cap_chat="results_capacity_chat_slots${slots}.json"
    local cap_mod="results_capacity_moderate_slots${slots}.json"
    local cap_batch="results_capacity_batch_slots${slots}.json"
    for f in "$cap_chat" "$cap_mod" "$cap_batch"; do
        if [[ ! -f "$f" ]]; then
            echo "  [stress] MISSING $f -- skipping $label"
            return 1
        fi
    done

    local extra_args=()
    case "$scenario" in
        1) extra_args=(--reserved-chat-slots 0) ;;
        2) extra_args=(--reserved-chat-slots 1) ;;
        3) extra_args=(--reserved-chat-slots 1 --smart-preemption) ;;
    esac

    echo "  [stress] running $label (duration=${DURATION}s, rate_frac=${RATE_FRAC}) ..."
    python3 continuous_stress.py \
        --scenario "$scenario" --slots "$slots" --duration "$DURATION" \
        --chat-capacity-json "$cap_chat" --moderate-capacity-json "$cap_mod" \
        --batch-capacity-json "$cap_batch" --rate-frac "$RATE_FRAC" \
        --drain-timeout "$DRAIN_TIMEOUT" \
        "${extra_args[@]}" \
        --label "$label" --output "$output" \
        2>&1 | tee "${OUTDIR}/log_${label}.txt"

    if [[ -f "$output" ]]; then
        echo "  [stress] OK -> $output"
    else
        echo "  [stress] FAILED -- no output written for $label (see log above)"
    fi
}

echo "=== rf022 matrix (ARCHIVED methodology): Scenarios 1-3 x slots {1,2,4,8} ==="
echo

for slots in "${SLOT_COUNTS[@]}"; do
    echo "### slots=$slots ###"
    if ! start_server "$slots"; then
        continue
    fi
    for scenario in "${SCENARIOS[@]}"; do
        echo " -- scenario $scenario, slots $slots --"
        if start_proxy "$scenario" "$slots"; then
            run_stress "$scenario" "$slots"
            stop_proxy
        fi
        sleep 1
    done
    stop_server
    echo
done

echo "=== Done. Results in ${OUTDIR}/ ==="