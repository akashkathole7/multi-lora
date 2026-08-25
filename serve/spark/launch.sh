#!/usr/bin/env bash
# Secondary serving path: the same two-tenant vLLM stack on an NVIDIA DGX Spark.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE QUOTING ANY NUMBER THIS PRODUCES
# ---------------------------------------------------------------------------
# This is the ON-PREM path. It exists so the project is not welded to Azure, and
# so a customer who wants the model inside their own building has a documented
# route. It is NOT the path the project's headline numbers come from.
#
# Every latency, throughput and memory figure produced against a DGX Spark is
# HARDWARE-DEPENDENT and is not comparable with the Azure A100 numbers:
#
#   Azure path   1x A100 80GB, dedicated HBM2e, x86_64 host, PCIe/SXM
#   Spark path   GB10 superchip: Blackwell GPU + Grace CPU (arm64) sharing ONE
#                128 GB unified memory pool
#
# Unified memory in particular changes the shape of the answer. On the A100 the
# 16 GB of base weights sit in dedicated GPU memory and nothing else competes
# for it. On the Spark, the OS, the container runtime, the KV cache and the
# model all draw on the same 128 GB. The official vLLM DGX Spark post is
# explicit: "--gpu-memory-utilization should leave headroom in the unified
# memory pool for the operating system, container runtime, and KV cache growth."
#   https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
#
# So: label any Spark result as "DGX Spark (GB10, unified memory)" wherever it
# appears, and never put it in the same table as an A100 result without that
# label. bench/*.py already record the endpoint they measured; keep it that way.
#
# ---------------------------------------------------------------------------
# THE ARM64 / GB10 IMAGE SITUATION - the honest version
# ---------------------------------------------------------------------------
# Verified 2026-08-24:
#
#   * vllm/vllm-openai DOES publish aarch64 tags. `v0.27.1-aarch64` and
#     `v0.27.1-aarch64-cu129` are both on Docker Hub, pushed 2026-08-11.
#     https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?name=v0.27
#
#   * An aarch64 tag is NOT the same thing as GB10 support. The DGX Spark's
#     GB10 is compute capability sm_121 and wants CUDA 13; the published
#     aarch64 release tags above are cu129 (CUDA 12.9). The official vLLM DGX
#     Spark post recommends `vllm/vllm-openai:cu130-nightly` instead, and warns
#     in the same breath: "nightly tags move over time, treat cu130-nightly as
#     a compatibility track rather than a reproducible pin."
#
#   * The vLLM Docker docs describe aarch64 as something you BUILD: "A docker
#     container can be built for aarch64 systems such as the Nvidia Grace-Hopper
#     and Grace-Blackwell" with `--platform "linux/arm64"`.
#     https://docs.vllm.ai/en/latest/deployment/docker.html
#
# CHECK: whether any PINNED, REPRODUCIBLE vLLM image tag supports GB10/sm_121
# CHECK: could not be confirmed. The evidence says: a moving nightly works, a
# CHECK: pinned release tag probably does not, and NVIDIA's own NGC vLLM
# CHECK: container is the other candidate. Resolve it ON THE BOX with
# CHECK: `--check-image`, which pulls the tag and prints the CUDA version, the
# CHECK: torch build and the GPU vLLM sees, then pin the digest and record it in
# CHECK: change_log.md. Do not benchmark against a moving tag: two runs a week
# CHECK: apart would not be measuring the same software.
#
# CHECK: DEFAULT_IMAGE below is set to the nightly the official post recommends,
# CHECK: which is a compatibility track and NOT reproducible. Override it with
# CHECK: IMAGE=<something you pinned> for any run whose numbers you intend to
# CHECK: publish.
#
# Usage:
#   ./serve/spark/launch.sh                       # start the server
#   ./serve/spark/launch.sh --check-image         # pull + report, start nothing
#   ADAPTER_DIR=/data/loras ./serve/spark/launch.sh
#   IMAGE=vllm/vllm-openai:v0.27.1-aarch64 ./serve/spark/launch.sh

set -uo pipefail

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { printf 'launch: FATAL %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# CHECK: moving tag. See the block above. Pin before publishing numbers.
DEFAULT_IMAGE="vllm/vllm-openai:cu130-nightly"
IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"

BASE_MODEL="${BASE_MODEL:-nvidia/Llama-3.1-Nemotron-Nano-8B-v1}"
SERVED_BASE_NAME="${SERVED_BASE_NAME:-base}"
ADAPTER_NAMES="${ADAPTER_NAMES:-meridian,vantage}"

# Local adapter directory: one subdirectory per adapter, HF PEFT layout. Same
# shape as the Azure model asset, so the --lora-modules line below is identical.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ADAPTER_DIR="${ADAPTER_DIR:-${REPO_ROOT}/train/out/adapters}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MAX_LORAS="${MAX_LORAS:-4}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-8}"

# Lower than the Azure default of 0.90 on purpose: on GB10 this fraction is
# taken out of the pool the OS is also living in. 0.85 is the value the official
# vLLM DGX Spark post uses.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# The same post uses --max-num-seqs 4, "keeping batch size low for unified-memory
# efficiency". Note this interacts with MAX_LORAS: a batch of 4 sequences can
# carry at most 4 distinct adapters, which happens to match. Raise both together
# or neither.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

CONTAINER_NAME="${CONTAINER_NAME:-multilora-vllm-spark}"
MEM_SAMPLE_SECONDS="${MEM_SAMPLE_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/serve/spark/logs}"

CHECK_IMAGE_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --check-image) CHECK_IMAGE_ONLY=1 ;;
        -h|--help)     sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             die "unknown argument: ${arg}" ;;
    esac
done

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"

# ---------------------------------------------------------------------------
# GPU memory logging - the same strategy as the Azure path, so the two are
# comparable as PROCEDURE even though the numbers are not comparable as VALUES.
# ---------------------------------------------------------------------------

sample_host_gpu() {
    local phase="$1" line
    while IFS= read -r line; do
        [ -z "${line}" ] && continue
        log "GPUMEM phase=${phase} ${line}"
    done < <(nvidia-smi --query-gpu=memory.used --format=csv 2>&1 || echo "nvidia-smi unavailable")
}

# ---------------------------------------------------------------------------
# --check-image: resolve the CHECK above on the actual box
# ---------------------------------------------------------------------------

if [ "${CHECK_IMAGE_ONLY}" -eq 1 ]; then
    log "pulling ${IMAGE} (this is the tag whose GB10 support is unverified)"
    docker pull "${IMAGE}" || die "could not pull ${IMAGE}"
    echo ""
    echo "--- image digest (PIN THIS) ---"
    docker inspect --format '{{index .RepoDigests 0}}' "${IMAGE}" 2>/dev/null \
        || echo "(no repo digest available)"
    echo ""
    echo "--- architecture ---"
    docker inspect --format '{{.Architecture}} / {{.Os}}' "${IMAGE}" 2>/dev/null || true
    echo ""
    echo "--- CUDA / torch / GPU as seen inside the container ---"
    docker run --rm --gpus all --entrypoint python3 "${IMAGE}" -c '
import torch
print("torch          ", torch.__version__)
print("torch cuda     ", torch.version.cuda)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    print("capability      sm_%d%d" % (major, minor))
    print("GB10/DGX Spark expects sm_121")
try:
    import vllm
    print("vllm           ", vllm.__version__)
except Exception as exc:
    print("vllm import FAILED:", exc)
' 2>&1 || log "WARNING the container could not report its GPU. That answers the image question in this file's header: this tag does not work here."
    echo ""
    log "record the digest and the sm_ number in change_log.md, then set IMAGE to the digest"
    exit 0
fi

# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

[ -d "${ADAPTER_DIR}" ] || die "no adapter directory at ${ADAPTER_DIR} (set ADAPTER_DIR)"

IFS=',' read -r -a ADAPTERS <<< "${ADAPTER_NAMES}"
LORA_MODULES=()
for name in "${ADAPTERS[@]}"; do
    if [ ! -f "${ADAPTER_DIR}/${name}/adapter_config.json" ]; then
        die "no adapter_config.json in ${ADAPTER_DIR}/${name}
       Convert a NeMo checkpoint first:
         python train/convert_to_hf.py --tenant ${name}
       or check an existing directory:
         python train/convert_to_hf.py --verify ${ADAPTER_DIR}/${name}"
    fi
    size="$(du -sb "${ADAPTER_DIR}/${name}" 2>/dev/null | cut -f1)"
    log "adapter ${name}: ${ADAPTER_DIR}/${name} (${size:-unknown} bytes)"
    # Container-side path, since ADAPTER_DIR is mounted at /adapters.
    LORA_MODULES+=("${name}=/adapters/${name}")
done

mkdir -p "${HF_CACHE}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------

log "image            ${IMAGE}"
log "base model       ${BASE_MODEL}"
log "adapters         ${LORA_MODULES[*]}"
log "port             ${PORT}"
log "gpu mem util     ${GPU_MEMORY_UTILIZATION}  (unified memory - see header)"
sample_host_gpu "before_container_start"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    log "removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Flags below are identical to serve/azure/start_server.sh except for the two
# unified-memory ones. That is the point: the same server, the same adapter
# names, the same client contract. eval/ and bench/ do not know which box they
# are pointed at.
#
# The upstream image's ENTRYPOINT is ["vllm", "serve"], so the model is passed
# POSITIONALLY as the first argument after the image name. `--model` is rejected
# by `vllm serve`.
VLLM_ARGS=("${BASE_MODEL}"
           --served-model-name "${SERVED_BASE_NAME}"
           --host 0.0.0.0
           --port 8000
           --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
           --max-num-seqs "${MAX_NUM_SEQS}"
           --enable-lora
           --max-lora-rank "${MAX_LORA_RANK}"
           --max-loras "${MAX_LORAS}"
           --max-cpu-loras "${MAX_CPU_LORAS}"
           --lora-modules "${LORA_MODULES[@]}")
if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

log "docker run ${CONTAINER_NAME}"
docker run -d \
    --name "${CONTAINER_NAME}" \
    --runtime nvidia --gpus all \
    --ipc=host \
    -p "${PORT}:8000" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${ADAPTER_DIR}:/adapters:ro" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    "${IMAGE}" \
    "${VLLM_ARGS[@]}" \
    || die "docker run failed. If the error mentions sm_121, an unsupported GPU
       or a missing kernel, that is the image CHECK in this file's header
       answering itself. Try: $0 --check-image"

log "container started. following startup until /health answers."

# ---------------------------------------------------------------------------
# wait for health, sampling memory as we go
# ---------------------------------------------------------------------------

HEALTH_URL="http://127.0.0.1:${PORT}/health"
ready=0
elapsed=0
next_sample=0
while [ "${elapsed}" -lt 1800 ]; do
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        log "container exited during startup. last 40 log lines:"
        docker logs --tail 40 "${CONTAINER_NAME}" 2>&1 || true
        die "vLLM did not start"
    fi
    if curl -fsS -o /dev/null --max-time 5 "${HEALTH_URL}" 2>/dev/null; then
        ready=1
        break
    fi
    if [ "${elapsed}" -ge "${next_sample}" ]; then
        sample_host_gpu "starting_${elapsed}s"
        next_sample=$((elapsed + MEM_SAMPLE_SECONDS))
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ "${ready}" -ne 1 ]; then
    log "WARNING /health did not answer within ${elapsed}s"
    docker logs --tail 40 "${CONTAINER_NAME}" 2>&1 || true
    exit 1
fi

log "/health answered after ~${elapsed}s"
sample_host_gpu "server_ready_base_resident"

# Per-adapter cold load, bracketed by memory samples. Same procedure as the
# Azure path, so the adapter footprint is measured the same way on both.
for name in "${ADAPTERS[@]}"; do
    sample_host_gpu "before_first_request_${name}"
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 300 \
        -H 'Content-Type: application/json' \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -d "{\"model\":\"${name}\",\"max_tokens\":1,\"messages\":[{\"role\":\"system\",\"content\":\"detailed thinking off\"},{\"role\":\"user\",\"content\":\"warm\"}]}" \
        2>/dev/null)"
    log "warm ${name} http=${code}"
    sample_host_gpu "after_first_request_${name}"
done

sample_host_gpu "steady_state"

cat <<EOF

  DGX Spark endpoint up on http://127.0.0.1:${PORT}
  served names: ${SERVED_BASE_NAME} ${ADAPTER_NAMES}

  Point the tools at it:
    python eval/separation.py --endpoint http://127.0.0.1:${PORT} \\
        --goals eval/goals_dev.jsonl
    python bench/swap_time.py --endpoint http://127.0.0.1:${PORT} --adapter meridian

  NOTE swap_time.py needs a COLD adapter. This script warmed both adapters
  above, so restart the container before measuring:
    docker rm -f ${CONTAINER_NAME} && $0

  Label every number from this box as DGX Spark (GB10, unified memory).
  It is not comparable with the Azure A100 numbers.

  Logs:     docker logs -f ${CONTAINER_NAME}
  Stop:     docker rm -f ${CONTAINER_NAME}
EOF
