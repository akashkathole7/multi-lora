# Serving on Azure ML

One A100, one frozen base model, two LoRA adapters, an OpenAI-compatible API.
Adapter selection is the `model` field on the request: `base`, `meridian`,
`vantage`.

> **Teardown is mandatory.** A `Standard_NC24ads_A100_v4` deployment bills from
> creation to deletion — roughly **₹320 / $3.67 per hour**, whether or not a
> single request arrives. Leaving one up for a weekend costs more than this
> project's entire ₹10,000 budget. `deploy.sh teardown` is step 2 of a 2-step
> process, not optional cleanup.

## Files

```
Dockerfile         vLLM base image + start_server.sh
start_server.sh    entrypoint: GPU memory logging, then vLLM with both adapters
environment.yaml   az ml environment: build context + inference_config routes
endpoint.yaml      az ml managed online endpoint (auth_mode: key)
deployment.yaml    az ml managed online deployment (SKU, probes, env vars)
deploy.sh          zero -> scoring URI -> smoke test -> teardown
```

## Command order

```bash
export AZ_BIN="$HOME/.venvs/azcli/bin/az"      # default; override if elsewhere
export SUBSCRIPTION_ID="<your subscription>"    # has a default, parameterised

./serve/azure/deploy.sh cost            # 1. see the bill before agreeing to it
./serve/azure/deploy.sh --hours 3       # 2. the whole thing
# ... measure ...
./serve/azure/deploy.sh teardown        # 3. STOP THE BILLING
```

`deploy.sh` with no subcommand runs, in order:

| # | step | time | money |
| --- | --- | --- | --- |
| 1 | stage adapters from `train/out/<tenant>_hf/` | seconds | none |
| 2 | **cost guardrail — type `yes-bill` or nothing happens** | — | none |
| 3 | `az account set` | seconds | none |
| 4 | resource group + workspace (idempotent) | 2–5 min first time | pennies/month |
| 5 | storage account + container + adapter upload | 1–2 min | pennies/month |
| 6 | register environment (ACR builds the Dockerfile) | 5–15 min first time | ACR build minutes |
| 7 | create endpoint | 1–3 min | **none** — an endpoint with no deployment does not bill |
| 8 | **create deployment** | 20–40 min | **billing starts here** |
| 9 | fetch scoring URI + key | seconds | — |
| 10 | probe which URL serves chat completions | seconds | — |
| 11 | smoke test, one completion per served name | ~1 min | — |
| 12 | print GPU memory lines from `get-logs` | seconds | — |

Everything above step 2 is free, and step 1 fails loudly if the adapters are
missing — so a missing adapter costs nothing rather than being discovered after
40 minutes of a billing A100.

Step 8 is slow because the container downloads ~16 GB of base weights at
startup. That is why the probes in `deployment.yaml` are set to ~35 minutes of
startup budget instead of the default ~5.

## The cost guardrail

Before any billable command, `deploy.sh` prints the SKU, the rate, the session
projection for `--hours N` (default 3), and refuses to continue until the
operator types `yes-bill` exactly. In a non-interactive shell it refuses
outright unless `ASSUME_YES=yes-bill` is set.

The rate ($3.673/hour) is a **hardcoded estimate** from the Azure pricing page
and is printed marked as such, with the URL to verify it at run time. It has not
been checked against a real invoice, a region, a currency or any credit.

## Teardown, and why it verifies

```bash
./serve/azure/deploy.sh teardown
```

Deletes the deployment, then the endpoint, then **lists the endpoints in the
workspace and checks the name is gone**. A delete command that returned zero is
not evidence that the resource is gone; a listing is. If the name is still
there, teardown exits nonzero with a loud block telling you that you are
probably still being billed and how to force it.

To remove everything including the workspace's storage, key vault and container
registry:

```bash
"$AZ_BIN" group delete --name multilora-rg --yes
```

## Routing — the one genuinely unresolved thing

A managed online endpoint's scoring URI ends in `/score`. This project's
container serves vLLM's `/v1/chat/completions`. `environment.yaml` bridges that
with `inference_config.scoring_route.path: /v1/chat/completions`, which is the
documented BYOC pattern — the Azure custom-container how-to does the same thing
with a TensorFlow Serving path.

What the Azure docs do **not** state is what literal path the container receives
when a client POSTs to `/score`, or whether a client may address
`/v1/chat/completions` on the public URI directly. Both readings are consistent
with the documentation.

Rather than guess, `deploy.sh` probes both against the live endpoint and prints
which one answered:

```
probe https://<ep>.eastus.inference.ml.azure.com/score -> HTTP 200
probe https://<ep>.eastus.inference.ml.azure.com/v1/chat/completions -> HTTP 404
CHAT COMPLETIONS URL: https://<ep>.eastus.inference.ml.azure.com/score
```

The winner is written to `serve/azure/logs/endpoint.env` as
`MULTILORA_CHAT_URL`. **Record the answer in `change_log.md`** — it is a fact
about Azure that this project had to discover, and the next person should not
have to spend A100 minutes rediscovering it.

Note the interaction with the measurement tools: `eval/separation.py` and
`bench/*.py` normalise `--endpoint` by appending `/v1/chat/completions` unless
it already ends in `/chat/completions`. A URL ending in `/score` is not
something they can construct, so if `/score` is the winner those tools need a
local path-rewriting proxy in front of them. If the `/v1` form wins, pass
`--endpoint https://<ep>.<region>.inference.ml.azure.com` and nothing changes.

## How the GPU memory numbers are captured

There is no API for "how much GPU memory did that adapter take". The mechanism
is stdout.

`start_server.sh` labels its strategy explicitly in its own header:
**external sampling + explicit phase markers.** vLLM exposes no hook that fires
when the base weights or an individual adapter finish loading, so the script
brackets the events it *can* observe from outside the server process:

| marker | when |
| --- | --- |
| `phase=startup` | before anything is loaded |
| `phase=after_base_download` | weights on disk (not yet on the GPU) |
| `phase=server_ready_base_resident` | `/health` answered — base weights are resident |
| `phase=before_first_request_<name>` | immediately before that adapter's cold load |
| `phase=after_first_request_<name>` | immediately after it |
| `phase=periodic` | every `MEM_SAMPLE_SECONDS` (default 30), always |
| `phase=steady_state` | after warm-up |

Every line is `nvidia-smi --query-gpu=memory.used --format=csv` output prefixed
with `GPUMEM`, timestamped, on stdout. Azure ML captures container stdout, and:

```bash
"$AZ_BIN" ml online-deployment get-logs \
    --name blue --endpoint-name multilora-ep \
    --resource-group multilora-rg --workspace-name multilora-ws \
    --container inference-server --lines 5000
```

replays it. `./serve/azure/deploy.sh logs` wraps that, saves the full log to
`serve/azure/logs/deployment_logs.txt`, and greps out the phase markers.

**The number that matters** is the delta between
`before_first_request_<name>` and `after_first_request_<name>`. That is the
adapter's GPU footprint, and it is what replaces the 0.08 GB estimate in
`bench/economics.py`.

One interaction to know about: `WARM_ADAPTERS=1` (the default) fires one
request per adapter at startup so real users never pay the cold load. That also
means the adapters are warm before `bench/swap_time.py` gets there. Set
`WARM_ADAPTERS: "0"` in `deployment.yaml` and redeploy before measuring cold
swap time.

## If startup times out

Symptom: the deployment fails while `get-logs` still shows the download
running. The probes in `deployment.yaml` allow ~35 minutes, but Azure ML may
impose its own provisioning timeout that is not documented.

The fix is to stop downloading at startup: bake the base weights into the image
(add a `RUN python3 -c "from huggingface_hub import snapshot_download; ..."`
layer to the `Dockerfile`), or register them as a second model asset. Both trade
a much larger image or a longer registration for a container that starts in
under a minute. Neither has been tried here.

## Adapters: blob storage vs model asset

`deploy.sh` uploads the adapters to an Azure Blob container in your
subscription — that is the system of record, and it is the "your data never
leaves your subscription" story in `ARCHITECTURE.md`.

Getting them *into* the container is a separate question. The documented
mechanism for a managed online deployment is a registered **model asset**,
mounted at `model_mount_path` and exposed as `AZUREML_MODEL_DIR`. No documented
way to mount a raw blob container directly into a managed online deployment was
found; blob/datastore mounting is documented for jobs, not for online
deployments. `start_server.sh` does not assume a fixed layout — it searches
`AZUREML_MODEL_DIR`, then `MODEL_MOUNT_PATH`, then `/mnt/adapters`, for a
directory containing the first adapter name, and logs which one it found.

If adapters need to change without a redeploy, the verified alternative is
vLLM's runtime LoRA API (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` plus
`POST /v1/load_lora_adapter`). Not used here: it widens the attack surface on a
key-auth endpoint for no benefit at two tenants.

## After it is up

```bash
source serve/azure/logs/endpoint.env

.venv/bin/python eval/separation.py --endpoint "$MULTILORA_CHAT_URL" \
    --api-key-env MULTILORA_API_KEY --goals eval/goals_sealed.jsonl --sealed
.venv/bin/python bench/swap_time.py --endpoint "$MULTILORA_CHAT_URL" --adapter meridian
.venv/bin/python bench/run_matrix.py --endpoint "$MULTILORA_CHAT_URL" --requests-per-arm 24
```

`endpoint.env` holds the bearer key and is written mode 600. It is gitignored.
No tool in this repo accepts a key as a command-line flag — only via an
environment variable named by `--api-key-env`.

Then, immediately:

```bash
./serve/azure/deploy.sh teardown
```
