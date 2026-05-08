#!/usr/bin/env bash
set -u
set -o pipefail

# ============================================================
# Run 8 Gemini experiments through Vertex AI OpenAI-compatible API.
# Each experiment is split into 6 index ranges. Before each range,
# the script refreshes the short-lived Google OAuth access token.
#
# Outputs:
#   results/gemini/<experiment_name>/
# ============================================================

# -------------------------
# User config
# -------------------------
PROJECT_ID="${PROJECT_ID:-gemini-vertex-translate}"
LOCATION="${LOCATION:-global}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results/gemini}"
CONCURRENCY="${CONCURRENCY:-2}"
RUNS="${RUNS:-1}"

VERTEX_OPENAI_BASE_URL="${VERTEX_OPENAI_BASE_URL:-https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/openapi}"

# Original AgentBench OS: 144 tasks, indices 0..143. Six ranges of 24.
OS_STD_RANGES=(
  "0-23"
  "24-47"
  "48-71"
  "72-95"
  "96-119"
  "120-143"
)

# HORIZON safe subset: 146 tasks, indices 0..145. Six near-balanced ranges.
HORIZON_RANGES=(
  "0-23"
  "24-47"
  "48-71"
  "72-95"
  "96-120"
  "121-145"
)

FLASH_MODEL="google/gemini-3-flash-preview"
PRO_MODEL="google/gemini-3.1-pro-preview"

# -------------------------
# Helpers
# -------------------------
mkdir -p "${BASE_RESULTS_DIR}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${BASE_RESULTS_DIR}/overnight_6seg_master_${RUN_TS}.log"
SUMMARY_FILE="${BASE_RESULTS_DIR}/overnight_6seg_summary_${RUN_TS}.tsv"

echo -e "timestamp\tmodel\ttask\trange\tstatus\toutput_dir" | tee -a "${SUMMARY_FILE}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

refresh_token() {
  log "Refreshing Google access token..."
  export OPENAI_API_KEY="$(gcloud auth print-access-token)"
}

run_segment() {
  local model="$1"
  local task="$2"
  local range="$3"
  local output_dir="$4"

  mkdir -p "${output_dir}"

  local safe_model_name
  safe_model_name="$(echo "${model}" | tr '/:' '__')"
  local safe_range
  safe_range="$(echo "${range}" | tr '-' '_')"
  local segment_log="${output_dir}/segment_${safe_model_name}_${task}_${safe_range}_$(date +%Y%m%d_%H%M%S).log"

  log "START model=${model} task=${task} range=${range} out=${output_dir}"

  refresh_token

  agentrl-eval \
    --no-interactive \
    -c http://localhost:5020/api \
    -u "${VERTEX_OPENAI_BASE_URL}" \
    -m "${model}" \
    --indices-range "${range}" \
    --concurrency "${CONCURRENCY}" \
    -n "${RUNS}" \
    -o "${output_dir}" \
    "${task}" 2>&1 | tee -a "${segment_log}"

  local rc=${PIPESTATUS[0]}

  if [[ ${rc} -eq 0 ]]; then
    log "DONE  model=${model} task=${task} range=${range}"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S')\t${model}\t${task}\t${range}\tOK\t${output_dir}" | tee -a "${SUMMARY_FILE}"
  else
    log "FAIL  model=${model} task=${task} range=${range} exit_code=${rc}"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S')\t${model}\t${task}\t${range}\tFAIL_${rc}\t${output_dir}" | tee -a "${SUMMARY_FILE}"
  fi

  # Cooldown to reduce burst/rate-limit/server pressure.
  sleep 10
}

run_experiment() {
  local model="$1"
  local exp_name="$2"
  local task="$3"
  local range_type="$4"

  local output_dir="${BASE_RESULTS_DIR}/${exp_name}"

  log "===== EXPERIMENT ${exp_name} task=${task} model=${model} ====="

  if [[ "${range_type}" == "std" ]]; then
    for range in "${OS_STD_RANGES[@]}"; do
      run_segment "${model}" "${task}" "${range}" "${output_dir}"
    done
  elif [[ "${range_type}" == "horizon" ]]; then
    for range in "${HORIZON_RANGES[@]}"; do
      run_segment "${model}" "${task}" "${range}" "${output_dir}"
    done
  else
    log "Unknown range_type=${range_type}"
    return 1
  fi

  log "===== FINISHED ${exp_name} ====="
}

# -------------------------
# Preflight
# -------------------------
log "PROJECT_ID=${PROJECT_ID}"
log "LOCATION=${LOCATION}"
log "VERTEX_OPENAI_BASE_URL=${VERTEX_OPENAI_BASE_URL}"
log "BASE_RESULTS_DIR=${BASE_RESULTS_DIR}"
log "CONCURRENCY=${CONCURRENCY}"
log "RUNS=${RUNS}"

if ! command -v gcloud >/dev/null 2>&1; then
  log "ERROR: gcloud not found"
  exit 1
fi

if ! command -v agentrl-eval >/dev/null 2>&1; then
  log "ERROR: agentrl-eval not found"
  exit 1
fi

refresh_token

# -------------------------
# Gemini 3 Flash Preview: 4 experiments
# -------------------------
run_experiment "${FLASH_MODEL}" "gemini3_flash_os_std" "os-std" "std"
run_experiment "${FLASH_MODEL}" "gemini3_flash_os_std_monitor" "os-std-monitor-replan" "std"
run_experiment "${FLASH_MODEL}" "gemini3_flash_horizon_baseline" "os-horizon-all" "horizon"
run_experiment "${FLASH_MODEL}" "gemini3_flash_horizon_monitor" "os-horizon-all-monitor-replan" "horizon"

# -------------------------
# Gemini 3.1 Pro Preview: 4 experiments
# -------------------------
run_experiment "${PRO_MODEL}" "gemini31_pro_os_std" "os-std" "std"
run_experiment "${PRO_MODEL}" "gemini31_pro_os_std_monitor" "os-std-monitor-replan" "std"
run_experiment "${PRO_MODEL}" "gemini31_pro_horizon_baseline" "os-horizon-all" "horizon"
run_experiment "${PRO_MODEL}" "gemini31_pro_horizon_monitor" "os-horizon-all-monitor-replan" "horizon"

log "ALL EXPERIMENTS FINISHED"
log "Summary file: ${SUMMARY_FILE}"
