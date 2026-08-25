#!/usr/bin/env bash
# Container entrypoint: start vLLM with both tenant LoRA adapters, and make the
# GPU memory story visible in stdout so `az ml online-deployment get-logs` can
# recover it afterwards.
#
# Everything this script prints goes to stdout on purpose. Azure ML captures
# container stdout and `az ml online-deployment get-logs --container
# inference-server` replays it (container choices `inference-server` and
# `storage-initializer` verified at
# https://learn.microsoft.com/en-us/cli/azure/ml/online-deployment). That is the
# only channel this project has for reading numbers off the GPU, so the log
# lines below are the measurement, not decoration.
#
# GPU MEMORY LOGGING STRATEGY (this is the label the README refers to):
#   Strategy = "external sampling + explicit phase markers".
#   vLLM does not expose a hook that fires when the base weights or an
#   individual LoRA adapter finish loading, so this script cannot instrument
#   those events from the inside. Instead it does two things:
#     1. Explicit phase markers. It samples `nvidia-smi` at named points it
#        CAN observe from outside the server process: before anything starts,
#        after the base weights are downloaded to disk, once /health answers
#        (which is after the base weights are on the GPU), and immediately
#        before and after the very first request to each adapter (which is that
#        adapter's cold load).
#     2. A background sampler. Every ${MEM_SAMPLE_SECONDS} seconds it prints a
#        timestamped memory sample, so anything that happens between the markers
#        still leaves a trace.
#   Every sample line is prefixed GPUMEM so the log is greppable.
#
# Flags passed to vLLM, and where each was verified (docs checked 2026-08-24):
#   --served-model-name   https://docs.vllm.ai/en/latest/configuration/engine_args.html
#   --enable-lora         same
#   --max-lora-rank       same  (documented default 16)
#   --max-loras           same  ("Max number of LoRAs in a single batch", default 1)
#   --max-cpu-loras       same  ("Must be >= than max_loras")
#   --max-model-len       same
#   --gpu-memory-utilization  same
#   --lora-modules        https://docs.vllm.ai/en/latest/features/lora.html
#                         (documented forms: `name=path`, and a JSON object)
#   --port / --host       https://docs.vllm.ai/en/latest/cli/serve.html
#   GET /health           https://docs.vllm.ai/en/latest/serving/online_serving/
# Beyond the docs: --served-model-name, --enable-lora, --max-lora-rank 16,
# --max-loras 4, --lora-modules name=path and --max-model-len 4096 all ran LIVE
# on vllm/vllm-openai:v0.27.1 against this base model in Azure ML job
# sad_line_6gslz8gljq (train/azureml/job_devmatrix.yaml). They are not guesses.
#
# The model is passed POSITIONALLY, not as --model. `vllm serve` rejects
# --model with "With `vllm serve`, you should provide the model as a positional
# argument or in a config file instead of via the `--model` option", and the
# option is slated for removal; the CLI docs only ever show the positional form.
# See https://github.com/vllm-project/vllm/pull/16691.
#
# LINE DISCIPLINE (change_log entry 10). Every command below sits on ONE
# physical line, including the array appends that build the vLLM argv. A shell
# line whose first token is a bare `-flag` is the exact shape of the bug that
# cost this project ~3.2 A100-hours, so none appear here and a check in
# data/logs/serve_adapt_checks.log asserts that.

set -uo pipefail

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "FATAL $*"; exit 1; }

# --------------------------------------------------------------------------
# configuration (every value overridable from deployment.yaml)
# --------------------------------------------------------------------------

BASE_MODEL="${BASE_MODEL:-nvidia/Llama-3.1-Nemotron-Nano-8B-v1}"
SERVED_BASE_NAME="${SERVED_BASE_NAME:-base}"
ADAPTER_NAMES="${ADAPTER_NAMES:-meridian,vantage}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MAX_LORAS="${MAX_LORAS:-4}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MEM_SAMPLE_SECONDS="${MEM_SAMPLE_SECONDS:-30}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"
WARM_ADAPTERS="${WARM_ADAPTERS:-1}"
HF_HOME="${HF_HOME:-/mnt/hfcache}"
export HF_HOME

# ADAPTER_ROOT is the directory that directly contains one subdirectory per
# adapter. Normally left empty and discovered below from the mounted model
# asset; set it explicitly to skip discovery entirely.
ADAPTER_ROOT="${ADAPTER_ROOT:-}"
ADAPTER_ROOT_PATTERN=""
# Mirrors deployment.yaml's model_mount_path. Searched as a candidate root
# because whether AZUREML_MODEL_DIR follows a custom model_mount_path is not
# stated in the Azure docs.
ADAPTER_MOUNT_ROOT="${ADAPTER_MOUNT_ROOT:-/mnt/adapters}"

# --------------------------------------------------------------------------
# GPU memory sampling
# --------------------------------------------------------------------------

gpu_mem() {
    # The exact query the project spec asks for. `|| true` because a container
    # scheduled without a GPU must still produce a log line rather than die.
    nvidia-smi --query-gpu=memory.used --format=csv 2>&1 || echo "nvidia-smi unavailable"
}

sample() {
    local phase="$1"
    local line
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        log "GPUMEM phase=${phase} ${line}"
    done < <(gpu_mem)
}

sampler_loop() {
    while true; do
        sample "periodic"
        sleep "${MEM_SAMPLE_SECONDS}"
    done
}

# --------------------------------------------------------------------------
# 0. startup sample, before anything is loaded
# --------------------------------------------------------------------------

log "start_server: multi-lora vLLM entrypoint"
log "start_server: base model        ${BASE_MODEL}"
log "start_server: served base name  ${SERVED_BASE_NAME}"
log "start_server: adapters          ${ADAPTER_NAMES}"
log "start_server: port              ${VLLM_PORT}"
log "start_server: HF_HOME           ${HF_HOME}"
log "start_server: memory strategy   external sampling + explicit phase markers"
sample "startup"

log "start_server: nvidia-smi topology / driver"
nvidia-smi 2>&1 | head -15 || log "start_server: nvidia-smi unavailable"

# --------------------------------------------------------------------------
# 1. locate the adapters in the mounted model asset
# --------------------------------------------------------------------------
#
# The deployment mounts the registered `adapters-both` custom_model asset. The
# Azure docs pin down the mount DIRECTORY:
#   "a model registered with the name my-model and version 1 is located on the
#    following path inside your deployed container:
#    /var/azureml-app/azureml-models/my-model/1"
#   and with model_mount_path set, "<model_mount_path>/<model-name>/<version>"
#   https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-container
# They do NOT pin down whether the registered folder's own name survives inside
# it. The TF Serving BYOC sample on that same page implies it does
# ($MODEL_BASE_PATH/half_plus_two); the model-specification concept page says
# AZUREML_MODEL_DIR "points to the folder containing the root of the model
# artifacts", which implies it might not.
#   https://learn.microsoft.com/en-us/azure/machine-learning/concept-online-deployment-model-specification
#
# So both shapes are searched, in this order, and the resolved directory and the
# pattern that matched are LOGGED. If neither shape is found anywhere, the
# script exits nonzero rather than starting a base-only server that would fail
# every tenant request 30 minutes and several hundred rupees later.

IFS=',' read -r -a ADAPTERS <<< "${ADAPTER_NAMES}"

first_adapter="${ADAPTERS[0]:-}"
[ -n "${first_adapter}" ] || die "ADAPTER_NAMES is empty"

CANDIDATE_ROOTS=("${AZUREML_MODEL_DIR:-}" "${ADAPTER_MOUNT_ROOT}" "/mnt/adapters" "/var/azureml-app/azureml-models")

try_root() {
    # try_root <root> <first-adapter>  -> sets ADAPTER_ROOT/ADAPTER_ROOT_PATTERN
    local root="$1" first="$2" hit=""
    [ -n "${root}" ] || return 1
    [ -d "${root}" ] || { log "start_server: candidate root ${root} does not exist"; return 1; }
    log "start_server: searching candidate root ${root}"
    if [ -f "${root}/${first}/adapter_config.json" ]; then
        ADAPTER_ROOT="${root}"
        ADAPTER_ROOT_PATTERN="<root>/${first}/adapter_config.json"
        return 0
    fi
    for hit in "${root}"/*/"${first}"/adapter_config.json; do
        [ -f "${hit}" ] || continue
        ADAPTER_ROOT="$(dirname "$(dirname "${hit}")")"
        ADAPTER_ROOT_PATTERN="<root>/*/${first}/adapter_config.json"
        return 0
    done
    for hit in "${root}"/*/*/"${first}"/adapter_config.json; do
        [ -f "${hit}" ] || continue
        ADAPTER_ROOT="$(dirname "$(dirname "${hit}")")"
        ADAPTER_ROOT_PATTERN="<root>/*/*/${first}/adapter_config.json"
        return 0
    done
    hit="$(find "${root}" -maxdepth 6 -type f -path "*/${first}/adapter_config.json" -print -quit 2>/dev/null)"
    if [ -n "${hit}" ]; then
        ADAPTER_ROOT="$(dirname "$(dirname "${hit}")")"
        ADAPTER_ROOT_PATTERN="find -maxdepth 6 -path '*/${first}/adapter_config.json'"
        return 0
    fi
    return 1
}

if [ -n "${ADAPTER_ROOT}" ]; then
    ADAPTER_ROOT_PATTERN="ADAPTER_ROOT set explicitly in the environment"
else
    for candidate in "${CANDIDATE_ROOTS[@]}"; do
        try_root "${candidate}" "${first_adapter}" && break
    done
fi

if [ -z "${ADAPTER_ROOT}" ]; then
    log "start_server: adapter discovery FAILED. Searched, in order:"
    log "start_server:   AZUREML_MODEL_DIR = ${AZUREML_MODEL_DIR:-unset}"
    log "start_server:   ADAPTER_MOUNT_ROOT = ${ADAPTER_MOUNT_ROOT}"
    log "start_server:   /mnt/adapters"
    log "start_server:   /var/azureml-app/azureml-models"
    log "start_server: for both <root>/${first_adapter}/adapter_config.json and"
    log "start_server: <root>/*/${first_adapter}/adapter_config.json (and two levels deeper)."
    log "start_server: Directory listing of AZUREML_MODEL_DIR, for the next person:"
    ls -laR "${AZUREML_MODEL_DIR:-/nonexistent}" 2>&1 | head -60
    die "no adapter directory found. The deployment mounts azureml:adapters-both@latest; either the asset is missing/empty or the mount layout changed. Refusing to serve a base-only endpoint that would 404 every tenant request."
fi

log "start_server: adapter root    ${ADAPTER_ROOT}"
log "start_server: matched pattern ${ADAPTER_ROOT_PATTERN}"

LORA_ARGS=()
LOADED_ADAPTERS=()
LORA_ARGS+=(--enable-lora)
LORA_ARGS+=(--max-lora-rank "${MAX_LORA_RANK}")
LORA_ARGS+=(--max-loras "${MAX_LORAS}")
LORA_ARGS+=(--max-cpu-loras "${MAX_CPU_LORAS}")
LORA_ARGS+=(--lora-modules)
for name in "${ADAPTERS[@]}"; do
    path="${ADAPTER_ROOT}/${name}"
    if [ ! -f "${path}/adapter_config.json" ]; then
        log "start_server: WARNING ${path}/adapter_config.json missing; skipping ${name}"
        continue
    fi
    size="$(du -sb "${path}" 2>/dev/null | cut -f1)"
    log "start_server: adapter ${name} at ${path} (${size:-unknown} bytes on disk)"
    LORA_ARGS+=("${name}=${path}")
    LOADED_ADAPTERS+=("${name}")
done
if [ "${#LOADED_ADAPTERS[@]}" -eq 0 ]; then
    die "adapter root ${ADAPTER_ROOT} resolved but not one of [${ADAPTER_NAMES}] has an adapter_config.json under it"
fi
if [ "${#LOADED_ADAPTERS[@]}" -ne "${#ADAPTERS[@]}" ]; then
    log "start_server: WARNING only ${#LOADED_ADAPTERS[@]} of ${#ADAPTERS[@]} adapters resolved: ${LOADED_ADAPTERS[*]}"
fi

# --------------------------------------------------------------------------
# 2. base model to disk, then sample
# --------------------------------------------------------------------------

mkdir -p "${HF_HOME}" 2>/dev/null || log "start_server: WARNING could not create ${HF_HOME}"

log "start_server: downloading base weights into ${HF_HOME} (this is the slow step)"
download_start="$(date +%s)"
python3 - "$BASE_MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
path = snapshot_download(repo_id=model_id)
print(f"start_server: snapshot_download -> {path}", flush=True)
PYEOF
download_rc=$?
download_end="$(date +%s)"
if [ "${download_rc}" -ne 0 ]; then
    log "start_server: WARNING snapshot_download exited ${download_rc}; letting vLLM"
    log "start_server:          fetch the weights itself. If the endpoint is on a"
    log "start_server:          private network this is where it will hang."
else
    log "start_server: base weights on disk after $((download_end - download_start))s"
fi
# Downloading puts weights on DISK, not on the GPU. This sample is here so the
# log distinguishes "download finished" from "weights are resident".
sample "after_base_download"

# --------------------------------------------------------------------------
# 3. background sampler, then start vLLM
# --------------------------------------------------------------------------

sampler_loop &
SAMPLER_PID=$!
log "start_server: memory sampler running every ${MEM_SAMPLE_SECONDS}s (pid ${SAMPLER_PID})"

VLLM_ARGS=("${BASE_MODEL}")
VLLM_ARGS+=(--served-model-name "${SERVED_BASE_NAME}")
VLLM_ARGS+=(--host "${VLLM_HOST}")
VLLM_ARGS+=(--port "${VLLM_PORT}")
VLLM_ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
VLLM_ARGS+=("${LORA_ARGS[@]}")
# VLLM_EXTRA_ARGS lets the operator add a flag without rebuilding the image.
# Deliberately unquoted word-splitting: it is a flag string, not a path.
# shellcheck disable=SC2206
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    EXTRA=(${VLLM_EXTRA_ARGS})
    VLLM_ARGS+=("${EXTRA[@]}")
fi

log "start_server: exec vllm serve ${VLLM_ARGS[*]}"
vllm serve "${VLLM_ARGS[@]}" &
VLLM_PID=$!

shutdown() {
    log "start_server: signal received, stopping"
    kill "${SAMPLER_PID}" 2>/dev/null || true
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    exit 0
}
trap shutdown TERM INT

# --------------------------------------------------------------------------
# 4. wait for /health, then sample
# --------------------------------------------------------------------------

HEALTH_URL="http://127.0.0.1:${VLLM_PORT}/health"
log "start_server: polling ${HEALTH_URL} for up to ${READY_TIMEOUT_SECONDS}s"
ready=0
elapsed=0
while [ "${elapsed}" -lt "${READY_TIMEOUT_SECONDS}" ]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "start_server: vLLM exited during startup"
        break
    fi
    if curl -fsS -o /dev/null --max-time 5 "${HEALTH_URL}" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ "${ready}" -eq 1 ]; then
    log "start_server: /health answered after ~${elapsed}s"
    # First point at which the base weights are certainly on the GPU.
    sample "server_ready_base_resident"
else
    log "start_server: WARNING /health did not answer within ${READY_TIMEOUT_SECONDS}s"
    sample "server_not_ready"
fi

# --------------------------------------------------------------------------
# 5. per-adapter cold load, sampled either side
# --------------------------------------------------------------------------
#
# This is the closest thing to a load hook that exists from outside the server:
# the first request naming an adapter is what forces that adapter onto the GPU,
# so a sample immediately before and immediately after it brackets the cold
# load. The delta is the adapter's GPU footprint, and it is what replaces the
# ESTIMATE in bench/economics.py.
#
# It also means the first REAL user request is not the one paying the cold-start
# cost. Set WARM_ADAPTERS=0 in deployment.yaml if you are about to run
# bench/swap_time.py, which needs a genuinely cold adapter to measure.

if [ "${ready}" -eq 1 ] && [ "${WARM_ADAPTERS}" = "1" ]; then
    for name in "${LOADED_ADAPTERS[@]}"; do
        sample "before_first_request_${name}"
        t0="$(date +%s%N)"
        code="$(curl -sS -o /tmp/warm_"${name}".json -w '%{http_code}' --max-time 300 -H 'Content-Type: application/json' -X POST "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" -d "{\"model\":\"${name}\",\"max_tokens\":1,\"messages\":[{\"role\":\"system\",\"content\":\"detailed thinking off\"},{\"role\":\"user\",\"content\":\"warm\"}]}" 2>/dev/null)"
        t1="$(date +%s%N)"
        log "start_server: warm ${name} http=${code} elapsed_ms=$(( (t1 - t0) / 1000000 ))"
        sample "after_first_request_${name}"
    done
    log "start_server: adapter warm-up done. The GPUMEM deltas around"
    log "start_server: before_first_request_* / after_first_request_* are the"
    log "start_server: per-adapter GPU footprint."
elif [ "${WARM_ADAPTERS}" != "1" ]; then
    log "start_server: WARM_ADAPTERS=0, adapters left cold for bench/swap_time.py"
fi

log "start_server: serving. models available: ${SERVED_BASE_NAME} ${LOADED_ADAPTERS[*]}"
sample "steady_state"

wait "${VLLM_PID}"
rc=$?
log "start_server: vllm exited ${rc}"
kill "${SAMPLER_PID}" 2>/dev/null || true
exit "${rc}"
