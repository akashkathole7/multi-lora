#!/usr/bin/env bash
# End-to-end: registered adapters -> Azure ML managed online endpoint -> scoring
# URI -> smoke test -> GPU memory numbers. And, just as importantly, back to
# zero.
#
#   ./serve/azure/deploy.sh                 # full deploy (asks for confirmation)
#   ./serve/azure/deploy.sh --hours 6       # same, with a 6-hour cost projection
#   ./serve/azure/deploy.sh cost            # print the cost estimate and exit
#   ./serve/azure/deploy.sh smoke           # re-run the smoke test only
#   ./serve/azure/deploy.sh logs            # re-print the GPU memory log lines
#   ./serve/azure/deploy.sh teardown        # delete deployment + endpoint, verify
#
# ---------------------------------------------------------------------------
# THE ONE THING TO KNOW
# ---------------------------------------------------------------------------
# A Standard_NC24ads_A100_v4 managed online deployment bills from the moment it
# is created until it is deleted. Not per request. Not while it is warm. From
# creation to deletion, including every hour it sits idle overnight. At roughly
# 320 rupees an hour, a forgotten endpoint costs more in a weekend than this
# project's entire 10,000 rupee budget.
#
# `deploy.sh teardown` is not optional cleanup. It is step 2 of a 2-step
# process, and this script nags about it accordingly.
#
# ---------------------------------------------------------------------------
# WHERE THE ADAPTERS COME FROM
# ---------------------------------------------------------------------------
# A registered Azure ML model asset, `adapters-both` - a folder holding
# meridian/ and vantage/ HF PEFT adapter directories, written by the training
# jobs and never round-tripped through a laptop.
#
# ---------------------------------------------------------------------------
# az CLI commands used, verified 2026-08-24, re-checked 2026-08-25
# ---------------------------------------------------------------------------
#   az account set --subscription
#       https://learn.microsoft.com/en-us/cli/azure/account
#   az group create --name --location
#       https://learn.microsoft.com/en-us/cli/azure/group
#   az ml workspace show / create --name --resource-group --location
#       https://learn.microsoft.com/en-us/cli/azure/ml/workspace
#   az ml model show --name --version    (read-only pre-flight)
#       https://learn.microsoft.com/en-us/cli/azure/ml/model
#   az ml environment create --file
#       https://learn.microsoft.com/en-us/cli/azure/ml/environment
#   az ml online-endpoint create --file
#   az ml online-endpoint show --query scoring_uri
#   az ml online-endpoint get-credentials --query primaryKey
#   az ml online-endpoint list / delete --yes
#       https://learn.microsoft.com/en-us/cli/azure/ml/online-endpoint
#   az ml online-deployment create --file --all-traffic
#   az ml online-deployment get-logs --lines --container
#   az ml online-deployment delete --yes
#       https://learn.microsoft.com/en-us/cli/azure/ml/online-deployment
#
# NO storage-account or blob-upload command appears in this script any more. An
# earlier draft created a storage account, uploaded the adapters from the local
# disk and registered a model asset pointing at the blob copy. The registered
# `adapters-both` asset replaces all three steps: the artifact the endpoint
# serves is the artifact the training job wrote, there is no SAS token to mint
# or leak, and the version number is citable in a report.
#
# `--all-traffic` is on `online-deployment create` and not in endpoint.yaml
# because the endpoint schema states you can't set `traffic` at creation time.

set -uo pipefail

# ---------------------------------------------------------------------------
# configuration - every value overridable from the environment
# ---------------------------------------------------------------------------

# The az binary. Not on PATH in this project's build environment; the CLI lives
# in its own virtualenv so it cannot disturb .venv.
AZ_BIN="${AZ_BIN:-$HOME/.venvs/azcli/bin/az}"

# Parameterised on purpose. Never let a subscription id be the thing that is
# hard to change. These defaults are the LIVE workspace this project trains in.
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-97540688-009b-4a05-bea2-5cdea4cfe222}"
LOCATION="${LOCATION:-southcentralus}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-multilora}"
WORKSPACE="${WORKSPACE:-mlw-multilora}"

ENDPOINT_NAME="${ENDPOINT_NAME:-multilora-ep}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-multilora-vllm}"
ENVIRONMENT_VERSION="${ENVIRONMENT_VERSION:-}"   # empty = let AzureML autogenerate

# The registered model asset holding both tenants' adapters. `@latest` is the
# documented "most recently created version" reference form:
# https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-core-syntax
ADAPTER_MODEL_NAME="${ADAPTER_MODEL_NAME:-adapters-both}"
ADAPTER_MODEL_REF="${ADAPTER_MODEL_REF:-azureml:${ADAPTER_MODEL_NAME}@latest}"

INSTANCE_TYPE="${INSTANCE_TYPE:-Standard_NC24ads_A100_v4}"

# Cost model. Both numbers are estimates and are labelled as such everywhere
# they are printed.
SKU_USD_PER_HOUR="${SKU_USD_PER_HOUR:-3.673}"
INR_PER_USD="${INR_PER_USD:-87.0}"
HOURS="${HOURS:-3}"

CONFIRM_PHRASE="yes-bill"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="${REPO_ROOT}/serve/azure"
VENV_PY="${VENV_PY:-${REPO_ROOT}/.venv/bin/python}"

TENANTS="${TENANTS:-meridian vantage}"

RENDER_DIR="${RENDER_DIR:-${REPO_ROOT}/serve/azure/.rendered}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/serve/azure/logs}"

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

log()  { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
step() { printf '\n=== %s ===\n' "$*"; }
die()  { printf 'deploy: FATAL %s\n' "$*" >&2; exit 1; }

rule() { printf '%s\n' "$(printf '=%.0s' $(seq 1 74))"; }

az() { "${AZ_BIN}" "$@"; }

ws_args() {
    printf '%s' "--resource-group ${RESOURCE_GROUP} --workspace-name ${WORKSPACE}"
}

require_az() {
    [ -x "${AZ_BIN}" ] || die "az not found or not executable at ${AZ_BIN}. Set AZ_BIN."
    az version >/dev/null 2>&1 || die "'${AZ_BIN} version' failed. Is the CLI installed?"
    if ! az extension show --name ml >/dev/null 2>&1; then
        die "the 'ml' az extension is not installed. Run: ${AZ_BIN} extension add -n ml"
    fi
}

render() {
    # render <src.yaml> <dst.yaml>  - substitute the __PLACEHOLDER__ tokens.
    local src="$1" dst="$2"
    mkdir -p "$(dirname "${dst}")"
    sed \
        -e "s|__ENDPOINT_NAME__|${ENDPOINT_NAME}|g" \
        -e "s|__DEPLOYMENT_NAME__|${DEPLOYMENT_NAME}|g" \
        -e "s|__ENVIRONMENT_NAME__|${ENVIRONMENT_NAME}|g" \
        -e "s|__ENVIRONMENT_VERSION__|${ENVIRONMENT_VERSION}|g" \
        -e "s|__ADAPTER_MODEL_REF__|${ADAPTER_MODEL_REF}|g" \
        -e "s|__HF_TOKEN__|${HF_TOKEN:-}|g" \
        "${src}" > "${dst}" || die "could not render ${src}"
    log "rendered ${src} -> ${dst}"
}

# ---------------------------------------------------------------------------
# THE COST GUARDRAIL - runs before any billable command
# ---------------------------------------------------------------------------

print_cost() {
    local usd_total inr_hr inr_total
    usd_total="$(awk -v r="${SKU_USD_PER_HOUR}" -v h="${HOURS}" 'BEGIN{printf "%.2f", r*h}')"
    inr_hr="$(awk -v r="${SKU_USD_PER_HOUR}" -v x="${INR_PER_USD}" 'BEGIN{printf "%.0f", r*x}')"
    inr_total="$(awk -v r="${SKU_USD_PER_HOUR}" -v x="${INR_PER_USD}" -v h="${HOURS}" 'BEGIN{printf "%.0f", r*x*h}')"

    rule
    echo "  COST ESTIMATE - READ BEFORE CONTINUING"
    rule
    printf '  SKU                     %s (1x A100 80GB, 24 vCPU)\n' "${INSTANCE_TYPE}"
    printf '  region                  %s\n' "${LOCATION}"
    printf '  instances               1\n'
    printf '  rate                    $%s / hour   (ESTIMATE - VERIFY AT RUN TIME)\n' "${SKU_USD_PER_HOUR}"
    printf '                          ~Rs %s / hour at Rs %s per USD\n' "${inr_hr}" "${INR_PER_USD}"
    printf '  planned session         %s hours\n' "${HOURS}"
    printf '  PROJECTED SESSION COST  $%s   (~Rs %s)\n' "${usd_total}" "${inr_total}"
    rule
    echo "  The rate above is a hardcoded estimate taken from the Azure pricing"
    echo "  page and has NOT been checked against your actual bill, your region,"
    echo "  your currency, or any reservation or credit you may hold. Verify at:"
    echo "    https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/"
    echo ""
    echo "  BILLING IS BY WALL-CLOCK TIME FROM DEPLOYMENT CREATION TO DELETION."
    echo "  Not per request. An idle endpoint costs exactly the same as a busy"
    echo "  one. If you walk away without running:"
    echo ""
    echo "      $0 teardown"
    echo ""
    printf '  you will be billed ~Rs %s per hour, indefinitely.\n' "${inr_hr}"
    rule
}

confirm_billing() {
    print_cost
    if [ "${ASSUME_YES:-}" = "${CONFIRM_PHRASE}" ]; then
        log "ASSUME_YES=${CONFIRM_PHRASE} set in the environment; proceeding without a prompt"
        return 0
    fi
    if [ ! -t 0 ]; then
        die "not an interactive terminal and ASSUME_YES=${CONFIRM_PHRASE} is not set; refusing to bill"
    fi
    echo ""
    printf 'Type exactly "%s" to create billable resources, anything else to abort: ' "${CONFIRM_PHRASE}"
    local answer=""
    read -r answer
    if [ "${answer}" != "${CONFIRM_PHRASE}" ]; then
        echo ""
        log "aborted. Nothing was created. No charge."
        exit 0
    fi
    echo ""
    log "confirmed. Creating billable resources."
    log "REMEMBER: $0 teardown"
}

# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

set_subscription() {
    step "az account set"
    az account set --subscription "${SUBSCRIPTION_ID}" || die "could not select subscription ${SUBSCRIPTION_ID}"
    az account show --query "{name:name, id:id}" -o tsv || true
}

check_model_asset() {
    # Read-only, free, and deliberately BEFORE the billing confirmation: a
    # missing model asset should cost nothing, not be discovered 30 minutes into
    # a running A100.
    step "check the adapter model asset exists (free, read-only)"
    local version
    # shellcheck disable=SC2046
    version="$(az ml model show --name "${ADAPTER_MODEL_NAME}" --label latest $(ws_args) --query version -o tsv 2>/dev/null)"
    if [ -z "${version}" ]; then
        rule
        echo "  Model asset '${ADAPTER_MODEL_NAME}' was not found in ${WORKSPACE}."
        echo ""
        echo "  It must be a custom_model folder laid out as:"
        echo "      <asset>/meridian/adapter_config.json + adapter_model.safetensors"
        echo "      <asset>/vantage/adapter_config.json  + adapter_model.safetensors"
        echo ""
        echo "  Register it with:"
        printf '      %s ml model create --name %s --type custom_model \\\n' "${AZ_BIN}" "${ADAPTER_MODEL_NAME}"
        printf '          --path <folder> %s\n' "$(ws_args)"
        rule
        die "no ${ADAPTER_MODEL_NAME} model asset. Nothing billable was created."
    fi
    log "model asset ${ADAPTER_MODEL_NAME}:${version} found; deployment will reference ${ADAPTER_MODEL_REF}"
    log "the adapters ship as a registered asset - no local staging, no blob upload, no SAS"
}

create_group_and_workspace() {
    step "resource group + workspace (idempotent - both already exist)"
    if az group show --name "${RESOURCE_GROUP}" >/dev/null 2>&1; then
        log "resource group ${RESOURCE_GROUP} already exists"
    else
        az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" -o none \
            || die "could not create resource group ${RESOURCE_GROUP}"
        log "created resource group ${RESOURCE_GROUP}"
    fi

    if az ml workspace show --name "${WORKSPACE}" --resource-group "${RESOURCE_GROUP}" >/dev/null 2>&1; then
        log "workspace ${WORKSPACE} already exists"
    else
        log "creating workspace ${WORKSPACE} (this also creates a storage account,"
        log "  key vault and container registry - a few rupees a month, not an A100)"
        az ml workspace create --name "${WORKSPACE}" --resource-group "${RESOURCE_GROUP}" \
            --location "${LOCATION}" -o none \
            || die "could not create workspace ${WORKSPACE}"
        log "created workspace ${WORKSPACE}"
    fi
}

register_environment() {
    step "register environment (BYOC image build)"
    render "${HERE}/environment.yaml" "${RENDER_DIR}/environment.yaml"
    # environment.yaml's build.path is `.`, resolved relative to the YAML file -
    # which is the rendered copy. Put exactly the two files the image needs next
    # to it, so the ACR build context is those two files and nothing else. In
    # particular it must not be serve/azure/, which contains logs/endpoint.env.
    cp "${HERE}/Dockerfile" "${RENDER_DIR}/Dockerfile" || die "could not stage Dockerfile"
    cp "${HERE}/start_server.sh" "${RENDER_DIR}/start_server.sh" || die "could not stage start_server.sh"
    chmod +x "${RENDER_DIR}/start_server.sh"
    log "build context ${RENDER_DIR}: Dockerfile + start_server.sh only"
    log "Azure ML builds the Dockerfile in the workspace container registry."
    log "  First build pulls the vLLM base image and takes several minutes."
    log "  Why a build at all: the deployment schema has no command/entrypoint"
    log "  override, and vllm serve takes its flags on ARGV - see the comment"
    log "  block in serve/azure/environment.yaml."
    # shellcheck disable=SC2046
    ENVIRONMENT_VERSION="$(az ml environment create --file "${RENDER_DIR}/environment.yaml" \
        $(ws_args) --query version -o tsv 2>/dev/null)"
    if [ -z "${ENVIRONMENT_VERSION}" ]; then
        die "environment create failed or returned no version"
    fi
    log "registered environment ${ENVIRONMENT_NAME}:${ENVIRONMENT_VERSION}"
}

create_endpoint() {
    step "create endpoint"
    render "${HERE}/endpoint.yaml" "${RENDER_DIR}/endpoint.yaml"
    # shellcheck disable=SC2046
    if az ml online-endpoint show --name "${ENDPOINT_NAME}" $(ws_args) >/dev/null 2>&1; then
        log "endpoint ${ENDPOINT_NAME} already exists"
    else
        # shellcheck disable=SC2046
        az ml online-endpoint create --file "${RENDER_DIR}/endpoint.yaml" $(ws_args) -o none \
            || die "endpoint create failed"
        log "created endpoint ${ENDPOINT_NAME}"
    fi
    log "an endpoint with no deployment does not bill. The next step does."
}

create_deployment() {
    step "create deployment  <-- THIS IS THE BILLABLE ONE"
    render "${HERE}/deployment.yaml" "${RENDER_DIR}/deployment.yaml"
    log "creating ${DEPLOYMENT_NAME} on ${INSTANCE_TYPE}, mounting ${ADAPTER_MODEL_REF}."
    log "  Expect 20-40 minutes: image build, node allocation, then ~16 GB of"
    log "  base weights downloaded inside the container. Billing starts now."
    # shellcheck disable=SC2046
    if az ml online-deployment show --name "${DEPLOYMENT_NAME}" --endpoint-name "${ENDPOINT_NAME}" $(ws_args) >/dev/null 2>&1; then
        log "deployment ${DEPLOYMENT_NAME} already exists - IT IS ALREADY BILLING. Reusing it."
        return 0
    fi
    # shellcheck disable=SC2046
    az ml online-deployment create --file "${RENDER_DIR}/deployment.yaml" $(ws_args) --all-traffic \
        || die "deployment create failed. Run '$0 teardown' - a failed deployment can still hold a node."
    log "deployment ${DEPLOYMENT_NAME} created and taking 100% of traffic"
}

fetch_credentials() {
    step "scoring URI + key"
    # shellcheck disable=SC2046
    SCORING_URI="$(az ml online-endpoint show --name "${ENDPOINT_NAME}" $(ws_args) \
        --query scoring_uri -o tsv 2>/dev/null)"
    # shellcheck disable=SC2046
    ENDPOINT_KEY="$(az ml online-endpoint get-credentials --name "${ENDPOINT_NAME}" $(ws_args) \
        --query primaryKey -o tsv 2>/dev/null)"
    [ -n "${SCORING_URI}" ] || die "could not read scoring_uri"
    [ -n "${ENDPOINT_KEY}" ] || die "could not read primaryKey"
    log "scoring URI: ${SCORING_URI}"
    log "key: retrieved (${#ENDPOINT_KEY} chars, not printed)"

    mkdir -p "${LOG_DIR}"
    {
        echo "# generated by deploy.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# source this to point the eval/ and bench/ tools at the endpoint"
        echo "export MULTILORA_SCORING_URI='${SCORING_URI}'"
        echo "export MULTILORA_API_KEY='${ENDPOINT_KEY}'"
    } > "${LOG_DIR}/endpoint.env"
    chmod 600 "${LOG_DIR}/endpoint.env"
    log "wrote ${LOG_DIR}/endpoint.env (mode 600, gitignored - it holds the key)"
}

# ---------------------------------------------------------------------------
# routing probe
# ---------------------------------------------------------------------------
# The Azure docs do not state what path a custom container receives when a
# client POSTs to the public scoring URI, nor whether a client may address an
# arbitrary path such as /v1/chat/completions on it. Rather than guess, probe
# both and report which one answers. The winner is what goes into --endpoint
# for eval/separation.py and bench/*.py.

CHAT_URL=""

probe_routes() {
    step "probe which URL actually serves chat completions"
    local base candidates url code
    base="${SCORING_URI%/score}"
    candidates="${SCORING_URI} ${base}/v1/chat/completions"
    for url in ${candidates}; do
        code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 120 \
            -H "Authorization: Bearer ${ENDPOINT_KEY}" \
            -H 'Content-Type: application/json' \
            -X POST "${url}" \
            -d '{"model":"base","max_tokens":1,"messages":[{"role":"system","content":"detailed thinking off"},{"role":"user","content":"ping"}]}' \
            2>/dev/null)"
        log "probe ${url} -> HTTP ${code}"
        if [ "${code}" = "200" ] && [ -z "${CHAT_URL}" ]; then
            CHAT_URL="${url}"
        fi
    done
    if [ -z "${CHAT_URL}" ]; then
        log "WARNING neither URL returned 200. Falling back to the scoring URI."
        CHAT_URL="${SCORING_URI}"
    fi
    rule
    echo "  CHAT COMPLETIONS URL: ${CHAT_URL}"
    echo ""
    echo "  This was DISCOVERED, not assumed - see the CHECK note in"
    echo "  serve/azure/environment.yaml. Record the answer in change_log.md."
    echo ""
    echo "  eval/ and bench/ normalise --endpoint by appending /v1/chat/completions"
    echo "  unless the URL already ends in /chat/completions. So:"
    if [ "${CHAT_URL}" = "${SCORING_URI}" ]; then
        echo "    the winning URL ends in /score, which those tools cannot build."
        echo "    Pass the URL below and they will use it verbatim only if it ends"
        echo "    in /chat/completions - it does not, so use a local rewrite proxy"
        echo "    or set scoring_route.path and call the /v1 form instead."
    else
        echo "    --endpoint '${SCORING_URI%/score}'"
    fi
    rule
    echo "export MULTILORA_CHAT_URL='${CHAT_URL}'" >> "${LOG_DIR}/endpoint.env"
}

# ---------------------------------------------------------------------------
# smoke test - one completion per served name, verdict from the local verifier
# ---------------------------------------------------------------------------

smoke_one() {
    # smoke_one <served-name> <tenant-contract-or-none>
    local model="$1" tenant="$2" body out code
    out="${LOG_DIR}/smoke_${model}.json"
    body="$(printf '{"model":"%s","max_tokens":2000,"stream":false,"messages":[{"role":"system","content":"detailed thinking off"},{"role":"user","content":"%s"}]}' \
        "${model}" "Cut unplanned downtime across our three sites by 30% within a year.")"

    code="$(curl -sS -o "${LOG_DIR}/smoke_${model}.raw.json" -w '%{http_code}' --max-time 180 \
        -H "Authorization: Bearer ${ENDPOINT_KEY}" \
        -H 'Content-Type: application/json' \
        -X POST "${CHAT_URL}" -d "${body}" 2>/dev/null)"

    if [ "${code}" != "200" ]; then
        printf '  %-9s HTTP %s  FAIL (no response to verify)\n' "${model}" "${code}"
        return 1
    fi

    # Pull choices[0].message.content out into a bare file for the verifier.
    "${VENV_PY}" -c '
import json, sys
raw, dest = sys.argv[1], sys.argv[2]
with open(raw, encoding="utf-8") as fh:
    payload = json.load(fh)
text = payload["choices"][0]["message"]["content"]
with open(dest, "w", encoding="utf-8") as fh:
    fh.write(text)
' "${LOG_DIR}/smoke_${model}.raw.json" "${out}" 2>/dev/null \
        || { printf '  %-9s HTTP 200  FAIL (response had no message content)\n' "${model}"; return 1; }

    if [ "${tenant}" = "none" ]; then
        # The base arm has no tenant contract to satisfy, so its smoke verdict is
        # the weakest useful one: did it return any text at all.
        local chars
        chars="$(wc -c < "${out}" 2>/dev/null | tr -d ' ')"
        if [ "${chars:-0}" -lt 1 ]; then
            printf '  %-9s HTTP 200  FAIL (returned an empty message)\n' "${model}"
            return 1
        fi
        printf '  %-9s HTTP 200  PASS (returns text, %s chars)\n' "${model}" "${chars}"
        echo "    context, NOT the smoke verdict - base is the control arm, and"
        echo "    the correct result below is FAIL under both tenant contracts:"
        local t
        for t in ${TENANTS}; do
            printf '      as %-9s ' "${t}"
            "${VENV_PY}" "${REPO_ROOT}/data/verifier.py" --tenant "${t}" "${out}" 2>&1 | head -1
        done
        return 0
    fi

    printf '  %-9s HTTP 200  ' "${model}"
    "${VENV_PY}" "${REPO_ROOT}/data/verifier.py" --tenant "${tenant}" "${out}" 2>&1 | head -1
}

smoke_test() {
    step "smoke test - one chat completion per served name"
    echo "Verdicts come from ${VENV_PY} data/verifier.py running LOCALLY on this"
    echo "machine, against the same deterministic contract that filtered the"
    echo "training data. Nothing on the Azure side decides whether this passed."
    echo ""
    local t rc=0
    for t in ${TENANTS}; do
        smoke_one "${t}" "${t}" || rc=1
    done
    smoke_one "base" "none" || rc=1
    echo ""
    if [ "${rc}" -eq 0 ]; then
        log "smoke test: every served name answered"
    else
        log "smoke test: at least one served name did not answer - see above"
    fi
    echo "Raw responses: ${LOG_DIR}/smoke_*.raw.json"
    echo "A one-request-per-arm smoke test is NOT the separation measurement."
    echo "For that, run eval/separation.py against the sealed goal set."
    return "${rc}"
}

# ---------------------------------------------------------------------------
# GPU memory numbers, straight out of container stdout
# ---------------------------------------------------------------------------

show_gpu_logs() {
    step "GPU memory log lines from the container"
    echo "Source: start_server.sh writes every nvidia-smi sample to stdout with a"
    echo "GPUMEM prefix; Azure ML captures container stdout; get-logs replays it."
    echo ""
    mkdir -p "${LOG_DIR}"
    local raw="${LOG_DIR}/deployment_logs.txt"
    # shellcheck disable=SC2046
    az ml online-deployment get-logs --name "${DEPLOYMENT_NAME}" \
        --endpoint-name "${ENDPOINT_NAME}" $(ws_args) \
        --container inference-server --lines 5000 > "${raw}" 2>/dev/null \
        || log "WARNING get-logs failed (deployment may not exist yet)"

    if [ -s "${raw}" ]; then
        log "wrote ${raw} ($(wc -l < "${raw}") lines)"
        echo ""
        echo "--- where the adapters were found inside the container ---"
        grep 'start_server: adapter root\|start_server: matched pattern' "${raw}" || echo "(none found)"
        echo ""
        echo "--- GPUMEM phase markers ---"
        grep 'GPUMEM phase=' "${raw}" | grep -v 'phase=periodic' || echo "(none found)"
        echo ""
        echo "--- adapter sizes on disk, as seen inside the container ---"
        grep 'start_server: adapter ' "${raw}" || echo "(none found)"
        echo ""
        echo "--- periodic samples (last 10) ---"
        grep 'phase=periodic' "${raw}" | tail -10 || echo "(none found)"
        echo ""
        echo "The delta between before_first_request_<name> and"
        echo "after_first_request_<name> is that adapter's GPU footprint. It is"
        echo "the number that replaces the 0.08 GB ESTIMATE in bench/economics.py."
    else
        log "no log output captured"
    fi
}

# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------

teardown() {
    require_az
    set_subscription

    local inr_hr
    inr_hr="$(awk -v r="${SKU_USD_PER_HOUR}" -v x="${INR_PER_USD}" 'BEGIN{printf "%.0f", r*x}')"

    step "teardown"
    log "deleting deployment ${DEPLOYMENT_NAME} (this is what stops the billing)"
    # shellcheck disable=SC2046
    az ml online-deployment delete --name "${DEPLOYMENT_NAME}" \
        --endpoint-name "${ENDPOINT_NAME}" $(ws_args) --yes -o none 2>/dev/null \
        || log "deployment delete returned nonzero (it may already be gone)"

    log "deleting endpoint ${ENDPOINT_NAME}"
    # shellcheck disable=SC2046
    az ml online-endpoint delete --name "${ENDPOINT_NAME}" $(ws_args) --yes -o none 2>/dev/null \
        || log "endpoint delete returned nonzero (it may already be gone)"

    step "VERIFY deletion"
    # A delete command that returned 0 is not evidence. Listing is.
    local remaining
    # shellcheck disable=SC2046
    remaining="$(az ml online-endpoint list $(ws_args) --query "[].name" -o tsv 2>/dev/null)"

    echo "endpoints still present in ${WORKSPACE}:"
    if [ -z "${remaining}" ]; then
        echo "  (none)"
    else
        printf '  %s\n' ${remaining}
    fi
    echo ""

    if printf '%s\n' ${remaining} | grep -qx "${ENDPOINT_NAME}"; then
        rule
        echo "  !!  TEARDOWN NOT CONFIRMED  !!"
        echo ""
        printf '  Endpoint %s IS STILL LISTED.\n' "${ENDPOINT_NAME}"
        printf '  If it still has a deployment you are STILL BEING BILLED at\n'
        printf '  ~Rs %s per hour.\n' "${inr_hr}"
        echo ""
        echo "  Deletion can lag by a minute or two - re-run:"
        printf '      %s teardown\n' "$0"
        echo "  and if it persists, delete it in the Azure portal by hand, or"
        echo "  delete the whole resource group:"
        printf '      %s group delete --name %s --yes\n' "${AZ_BIN}" "${RESOURCE_GROUP}"
        rule
        return 1
    fi

    rule
    echo "  TEARDOWN CONFIRMED"
    echo ""
    printf '  Endpoint %s is gone from the workspace listing.\n' "${ENDPOINT_NAME}"
    echo "  GPU billing for this endpoint has stopped."
    echo ""
    echo "  Still costing a little (rupees per month, not per hour):"
    echo "    - the workspace's storage account, key vault and container registry"
    printf '    - the registered model assets (%s and the per-tenant ones)\n' "${ADAPTER_MODEL_NAME}"
    echo "  To remove everything including those:"
    printf '      %s group delete --name %s --yes\n' "${AZ_BIN}" "${RESOURCE_GROUP}"
    echo ""
    printf '  REMINDER: an idle A100 endpoint bills ~Rs %s per hour. It does not\n' "${inr_hr}"
    echo "  matter that no requests were sent. Never leave one up overnight."
    rule
    return 0
}

# ---------------------------------------------------------------------------
# full deploy
# ---------------------------------------------------------------------------

deploy_all() {
    require_az
    set_subscription         # free
    check_model_asset        # free, and fails early if adapters-both is missing
    confirm_billing          # <-- nothing billable happens above this line
    create_group_and_workspace
    register_environment
    create_endpoint
    create_deployment
    fetch_credentials
    probe_routes
    smoke_test || true
    show_gpu_logs

    local inr_hr
    inr_hr="$(awk -v r="${SKU_USD_PER_HOUR}" -v x="${INR_PER_USD}" 'BEGIN{printf "%.0f", r*x}')"
    rule
    echo "  DEPLOYED - AND NOW BILLING"
    rule
    printf '  scoring URI   %s\n' "${SCORING_URI}"
    printf '  chat URL      %s\n' "${CHAT_URL}"
    printf '  adapters      %s\n' "${ADAPTER_MODEL_REF}"
    printf '  credentials   %s\n' "${LOG_DIR}/endpoint.env"
    echo ""
    echo "  Next:"
    printf '    source %s\n' "${LOG_DIR}/endpoint.env"
    echo "    ${VENV_PY} eval/separation.py --endpoint \"\$MULTILORA_CHAT_URL\" \\"
    echo "        --api-key-env MULTILORA_API_KEY --goals eval/goals_sealed.jsonl --sealed"
    echo "    ${VENV_PY} bench/swap_time.py --endpoint \"\$MULTILORA_CHAT_URL\" --adapter meridian"
    echo ""
    printf '  WHEN YOU ARE DONE - every hour from now costs ~Rs %s:\n' "${inr_hr}"
    printf '      %s teardown\n' "$0"
    rule
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

SUBCOMMAND="deploy"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --hours)      HOURS="${2:-3}"; shift 2 ;;
        --hours=*)    HOURS="${1#*=}"; shift ;;
        -h|--help)    usage; exit 0 ;;
        deploy|teardown|smoke|logs|cost)
                      SUBCOMMAND="$1"; shift ;;
        *)            die "unknown argument: $1  (try --help)" ;;
    esac
done

case "${SUBCOMMAND}" in
    cost)
        print_cost
        ;;
    deploy)
        deploy_all
        ;;
    teardown)
        teardown
        ;;
    smoke)
        require_az
        set_subscription
        fetch_credentials
        probe_routes
        smoke_test
        ;;
    logs)
        require_az
        set_subscription
        show_gpu_logs
        ;;
    *)
        die "unknown subcommand ${SUBCOMMAND}"
        ;;
esac
