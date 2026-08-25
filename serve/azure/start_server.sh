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
# Flags passed to vLLM, and where each was verified (all checked 2026-08-24):
#   --served-model-name   https://docs.vllm.ai/en/latest/configuration/engine_args.html
#   --enable-lora         same
#   --max-lora-rank       same  (documented default 16)
#   --max-loras           same  ("Max number of LoRAs in a single batch", default 1)
#   --max-cpu-loras       same  ("Must be >= than max_loras")
#   --gpu-memory-utilization  same
#   --lora-modules        https://docs.vllm.ai/en/latest/features/lora.html
#                         (documented forms: `name=path`, and a JSON object)
#   --port / --host / --api-key  https://docs.vllm.ai/en/latest/cli/serve.html
#   GET /health           https://docs.vllm.ai/en/latest/serving/online_serving/
#
# The model is passed POSITIONALLY, not as --model. `vllm serve` rejects
# --model with "With `vllm serve`, you should provide the model as a positional
# argument or in a config file instead of via the `--model` option", and the
# option is slated for removal; the CLI docs only ever show the positional form.
# See https://github.com/vllm-project/vllm/pull/16691.

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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MEM_SAMPLE_SECONDS="${MEM_SAMPLE_SECONDS:-30}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"
WARM_ADAPTERS="${WARM_ADAPTERS:-1}"
HF_HOME="${HF_HOME:-/mnt/hfcache}"
export HF_HOME

# ADAPTER_ROOT is the directory that directly contains one subdirectory per
# adapter. On Azure ML it is normally derived from the mounted model asset.
ADAPTER_ROOT="${ADAPTER_ROOT:-}"

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
# 1. locate the adapters
# --------------------------------------------------------------------------

IFS=',' read -r -a ADAPTERS <<< "${ADAPTER_NAMES}"

first_adapter="${ADAPTERS[0]:-}"
[ -n "${first_adapter}" ] || die "ADAPTER_NAMES is empty"

if [ -z "${ADAPTER_ROOT}" ]; then
    # Candidate roots, most specific first.
    #   AZUREML_MODEL_DIR is set by Azure ML in custom containers too; the
    #   registered model folder lands under it. Verified at
    #   https://learn.microsoft.com/en-us/azure/machine-learning/concept-online-deployment-model-specification
    #   and in the BYOC multimodel sample README.
    #   MODEL_MOUNT_PATH mirrors deployment.yaml's model_mount_path, where the
    #   model appears at <mount>/<model-name>/<version>/.
    for candidate in \
        "${AZUREML_MODEL_DIR:-}" \
        "${MODEL_MOUNT_PATH:-}" \
        "/mnt/adapters" \
        "/var/azureml-app/azureml-models"
    do
        [ -n "${candidate}" ] || continue
        [ -d "${candidate}" ] || continue
        if [ -d "${candidate}/${first_adapter}" ]; then
            ADAPTER_ROOT="${candidate}"
            break
        fi
        # The model asset is nested as <mount>/<model-name>/<version>/<adapter>.
        found="$(find "${candidate}" -maxdepth 4 -type d -name "${first_adapter}" -print -quit 2>/dev/null)"
        if [ -n "${found}" ]; then
            ADAPTER_ROOT="$(dirname "${found}")"
            break
        fi
    done
fi

if [ -z "${ADAPTER_ROOT}" ]; then
    log "start_server: WARNING no adapter root found. Searched AZUREML_MODEL_DIR"
    log "start_server:          (${AZUREML_MODEL_DIR:-unset}), MODEL_MOUNT_PATH"
    log "start_server:          (${MODEL_MOUNT_PATH:-unset}), /mnt/adapters and"
    log "start_server:          /var/azureml-app/azureml-models."
    log "start_server:          Starting base-only; adapter requests will 404."
else
    log "start_server: adapter root    ${ADAPTER_ROOT}"
fi

LORA_ARGS=()
LOADED_ADAPTERS=()
if [ -n "${ADAPTER_ROOT}" ]; then
    LORA_ARGS+=(--enable-lora
                --max-lora-rank "${MAX_LORA_RANK}"
                --max-loras "${MAX_LORAS}"
                --max-cpu-loras "${MAX_CPU_LORAS}"
                --lora-modules)
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
        log "start_server: WARNING no usable adapters found; starting base-only"
        LORA_ARGS=()
    fi
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

VLLM_ARGS=("${BASE_MODEL}"
           --served-model-name "${SERVED_BASE_NAME}"
           --host "${VLLM_HOST}"
           --port "${VLLM_PORT}"
           --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [ "${#LORA_ARGS[@]}" -gt 0 ]; then
    VLLM_ARGS+=("${LORA_ARGS[@]}")
fi
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

if [ "${ready}" -eq 1 ] && [ "${WARM_ADAPTERS}" = "1" ] && [ "${#LOADED_ADAPTERS[@]}" -gt 0 ]; then
    for name in "${LOADED_ADAPTERS[@]}"; do
        sample "before_first_request_${name}"
        t0="$(date +%s%N)"
        code="$(curl -sS -o /tmp/warm_"${name}".json -w '%{http_code}' \
            --max-time 300 \
            -H 'Content-Type: application/json' \
            -X POST "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
            -d "{\"model\":\"${name}\",\"max_tokens\":1,\"messages\":[{\"role\":\"system\",\"content\":\"detailed thinking off\"},{\"role\":\"user\",\"content\":\"warm\"}]}" \
            2>/dev/null)"
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

log "start_server: serving. models available: ${SERVED_BASE_NAME} ${LOADED_ADAPTERS[*]:-}"
sample "steady_state"

wait "${VLLM_PID}"
rc=$?
log "start_server: vllm exited ${rc}"
kill "${SAMPLER_PID}" 2>/dev/null || true
exit "${rc}"
