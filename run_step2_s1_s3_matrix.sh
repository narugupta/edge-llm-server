#!/usr/bin/env bash
# run_step2_s1_s3_matrix.sh — full Step 2 matrix for Scenarios 1-3
# Scenario 1: Priority proxy
# Scenario 2: Fixed reservation
# Scenario 3: Smart preemption
#
# For each slots in {1,2,4,8}:
#   - restart llama-server with --parallel matching slots
#   - for each scenario in {1,2,3}:
#       - start proxy.py with the flags that DEFINE that scenario
#       - run continuous_stress.py in --target-samples mode: aims for
#         TARGET_SAMPLES completed requests PER TYPE (chat/moderate/batch),
#         scaling all three down together (never singling one out) only if
#         that would exceed --combined-budget. Duration is set PER SLOT
#         COUNT below -- lower slots need much longer runs because batch's
#         mean service time (~50-58s) barely fits any request budget when
#         only 1-2 physical slots exist. See DURATION_BY_SLOTS.
#       - stop proxy.py
#   - stop llama-server
#
# Requires: results_capacity_{chat,moderate,batch}_slots{1,2,4,8}.json
# present in the working directory (from capacity-search.zip).
#
# Usage: ./run_step2_s1_s3_matrix.sh
# Results land in results/step2/results_step2_scenarioN_slotsS_ts15.json
#
# TOTAL RUNTIME: ~2 hours for the full 12-run matrix (dominated by slots=1
# needing ~21 min/run to get 15 batch samples). Run overnight, or lower
# TARGET_SAMPLES below (fewer samples = shorter but noisier results).

set -uo pipefail  # NOT -e: one failed run should not kill the whole matrix

MODEL="${MODEL:-$HOME/Qwen2.5-1.5B-Q4_K_M.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./build/bin/llama-server}"
SERVER_PORT=8081
PROXY_PORT=8082
TARGET_SAMPLES=15     # aim for ~15 completed requests PER TYPE per run --
                      # enough for a rough mean/violation-rate comparison
                      # without multi-hour runs. Bump to 20-30 for tighter
                      # confidence if you have the time (see the docstring
                      # above for the runtime tradeoff).
COMBINED_BUDGET=0.7   # ceiling on summed utilization across chat+moderate+
                      # batch sharing the same slots (must stay <1.0)
# Duration needed to hit TARGET_SAMPLES=15 for the SLOWEST type (batch) at
# each slot count, computed from mean_service_time in the Step-1 capacity
# JSONs: duration = target_n * sum(mean_service_time) / (slots * budget).
# Recompute these if you change TARGET_SAMPLES or COMBINED_BUDGET, or if
# you regenerate the capacity JSONs with different numbers.
declare -A DURATION_BY_SLOTS=( [1]=1250 [2]=620 [4]=380 [8]=210 )
DRAIN_TIMEOUT=180
WARMUP_SECONDS=5      # wait after starting llama-server before hammering it
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
    local duration="${DURATION_BY_SLOTS[$slots]}"
    if [[ -z "$duration" ]]; then
        echo "  [stress] no DURATION_BY_SLOTS entry for slots=$slots -- skipping"
        return 1
    fi
    local label="step2_scenario${scenario}_slots${slots}_ts${TARGET_SAMPLES}"
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

    # Extra CLI args just label the run for bookkeeping -- must match what
    # proxy.py was ACTUALLY started with above (continuous_stress.py does
    # not enforce this).
    local extra_args=()
    case "$scenario" in
        1) extra_args=(--reserved-chat-slots 0) ;;
        2) extra_args=(--reserved-chat-slots 1) ;;
        3) extra_args=(--reserved-chat-slots 1 --smart-preemption) ;;
    esac

    echo "  [stress] running $label (duration=${duration}s ~$((duration/60))min, target_samples=${TARGET_SAMPLES}) ..."
    python3 continuous_stress.py \
        --scenario "$scenario" --slots "$slots" --duration "$duration" \
        --chat-capacity-json "$cap_chat" --moderate-capacity-json "$cap_mod" \
        --batch-capacity-json "$cap_batch" \
        --target-samples "$TARGET_SAMPLES" --combined-budget "$COMBINED_BUDGET" \
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

echo "=== Step 2 matrix: Scenarios 1-3 x slots {1,2,4,8} ==="
echo "(S4 EDF, S5 WFQ+Preemption skipped -- not implemented in proxy.py yet)"
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
echo "Next: inspect ${OUTDIR}/results_step2_scenario*_slots*.json for"
echo "violation rates, SLO compliance, and [DEGRADING OVER RUN] flags."
