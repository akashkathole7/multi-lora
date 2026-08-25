# Multi-LoRA serving container for an Azure ML managed online endpoint (BYOC).
#
# Base image: the official vLLM OpenAI-compatible server image.
#   Tag v0.27.1 verified on Docker Hub 2026-08-24; the tag was pushed
#   2026-08-11. Tag list:
#   https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?name=v0.27
#   Docker usage docs: https://docs.vllm.ai/en/latest/deployment/docker.html
#
# Pin the tag. Do not use :latest here. An endpoint that silently changes vLLM
# version between deployments makes every benchmark in bench/ incomparable.
#
# The upstream image sets ENTRYPOINT ["vllm", "serve"] (verified in
# https://raw.githubusercontent.com/vllm-project/vllm/main/docker/Dockerfile,
# stage `vllm-openai`). We replace it, because we need to log GPU memory before
# and after the model load and Azure ML gives us no other hook to do that.
ARG VLLM_IMAGE_TAG=v0.27.1
FROM vllm/vllm-openai:${VLLM_IMAGE_TAG}

# Azure ML's managed online endpoint runs the container as-is and probes it over
# HTTP. curl is used by start_server.sh to poll vLLM's /health during startup.
# The upstream image is Ubuntu-based; install only if missing so a base-image
# change that already ships curl does not fail the build.
USER root
RUN if ! command -v curl >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends curl && \
        rm -rf /var/lib/apt/lists/*; \
    fi

COPY start_server.sh /opt/multilora/start_server.sh
RUN chmod +x /opt/multilora/start_server.sh

# Defaults. Every one of these is overridable from the deployment YAML's
# environment_variables block, so nothing about the tenant set is baked in.
ENV BASE_MODEL="nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
    SERVED_BASE_NAME="base" \
    ADAPTER_NAMES="meridian,vantage" \
    VLLM_PORT="8000" \
    MAX_LORA_RANK="16" \
    MAX_LORAS="4" \
    MAX_CPU_LORAS="8" \
    GPU_MEMORY_UTILIZATION="0.90" \
    MEM_SAMPLE_SECONDS="30" \
    HF_HOME="/mnt/hfcache"

EXPOSE 8000

ENTRYPOINT ["/opt/multilora/start_server.sh"]
