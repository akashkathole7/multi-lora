# Serving on Azure ML

One A100, one frozen base model, two LoRA adapters, an OpenAI-compatible API.
Adapter selection is the `model` field on the request: `base`, `meridian`,
`vantage`.

> **Teardown is mandatory.** A `Standard_NC24ads_A100_v4` deployment bills from
> creation to deletion — roughly **₹320 / $3.67 per hour**, whether or not a
> single request arrives. Leaving one up for a weekend costs more than this
> project's entire ₹10,000 budget. `deploy.sh teardown` is step 2 of a 2-step
> process, not optional cleanup.

Target workspace: **`mlw-multilora`** / **`rg-multilora`** / **southcentralus** —
the same workspace the adapters were trained in. Those are the defaults in
`deploy.sh`; every one is overridable from the environment.

## Files

```
Dockerfile         vLLM v0.27.1 + start_server.sh  (one thin layer, see below)
start_server.sh    entrypoint: find the mounted adapters, GPU memory logging,
                   then vLLM with both adapters
environment.yaml   az ml environment: build context + inference_config routes
endpoint.yaml      az ml managed online endpoint (auth_mode: key)
deployment.yaml    az ml managed online deployment (model asset, SKU, probes,
                   env vars)
deploy.sh          model-asset check -> scoring URI -> smoke test -> teardown
```

## Where the adapters come from

**A registered Azure ML model asset — `adapters-both` — not a blob upload.**

```
adapters-both:1   (custom_model)
├── meridian/     adapter_config.json + adapter_model.safetensors + tokenizer files
└── vantage/      adapter_config.json + adapter_model.safetensors + tokenizer files
```

The earlier draft of this directory staged the adapters into a local folder,
created a storage account, uploaded them with `az storage blob upload-batch`,
and then registered a model asset pointing at the blob copy. All three steps are
gone. The asset is written directly by the training jobs, so: **no laptop round
trip** (the bytes that get served are the bytes that were trained and evaluated,
never re-uploaded from a workstation), **no SAS token** to mint, rotate or leak,
and **a version number** — `adapters-both:1` — that a report can cite.

`deployment.yaml` references it as `azureml:adapters-both@latest`, the
documented "most recently created version" form
([core YAML syntax](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-core-syntax)).
`train/azureml/job_devmatrix.yaml` already uses the same form live against this
workspace.

### Where it lands inside the container, and why nothing assumes

Azure's docs pin down the mount **directory**: a model registered as `my-model`
version `1` appears at `/var/azureml-app/azureml-models/my-model/1`, or at
`<model_mount_path>/<model-name>/<version>` when `model_mount_path` is set
([custom container how-to](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-container)).
What they do **not** pin down for a folder-shaped `custom_model` is whether the
registered folder's own name survives inside that directory — the TF Serving
sample on that page implies it does, the
[model specification page](https://learn.microsoft.com/en-us/azure/machine-learning/concept-online-deployment-model-specification)
implies it may not. `adapters-both:1` was registered from a job output folder
literally named `adapters`, so this is not academic:

```
$AZUREML_MODEL_DIR/adapters/meridian/adapter_config.json     # folder name kept
$AZUREML_MODEL_DIR/meridian/adapter_config.json              # folder name dropped
```

`start_server.sh` handles **both**. It walks a fixed candidate list —
`$AZUREML_MODEL_DIR`, then `$ADAPTER_MOUNT_ROOT` (mirrors `model_mount_path`),
then `/mnt/adapters`, then `/var/azureml-app/azureml-models` — and at each root
tries `<root>/meridian/adapter_config.json`, then the glob
`<root>/*/meridian/adapter_config.json`, then one level deeper, then a bounded
`find`. It logs the directory it resolved and the pattern that matched:

```
start_server: adapter root    /mnt/adapters/adapters-both/1/adapters
start_server: matched pattern <root>/*/*/meridian/adapter_config.json
```

If nothing matches it dumps a recursive listing of `$AZUREML_MODEL_DIR` and
**exits nonzero**. It does not fall back to a base-only server: that would pass
the health probe, take 100% of traffic, and 404 every tenant request on a
billing A100.

## Command order

```bash
export AZ_BIN="$HOME/.venvs/azcli/bin/az"      # default; override if elsewhere

./serve/azure/deploy.sh cost            # 1. see the bill before agreeing to it
./serve/azure/deploy.sh --hours 3       # 2. the whole thing
# ... measure ...
./serve/azure/deploy.sh teardown        # 3. STOP THE BILLING
```

`deploy.sh` with no subcommand runs, in order:

| # | step | time | money |
| --- | --- | --- | --- |
| 1 | `az account set` | seconds | none |
| 2 | `az ml model show adapters-both` — fail early if the asset is missing | seconds | none |
| 3 | **cost guardrail — type `yes-bill` or nothing happens** | — | none |
| 4 | resource group + workspace, `show \|\| create` (both already exist) | seconds | pennies/month |
| 5 | register environment (ACR builds the Dockerfile) | 5–15 min first time | ACR build minutes |
| 6 | create endpoint | 1–3 min | **none** — an endpoint with no deployment does not bill |
| 7 | **create deployment**, mounting `azureml:adapters-both@latest` | 20–40 min | **billing starts here** |
| 8 | fetch scoring URI + key | seconds | — |
| 9 | probe which URL serves chat completions | seconds | — |
| 10 | smoke test, one completion per served name | ~1 min | — |
| 11 | print GPU memory lines from `get-logs` | seconds | — |
| 12 | print the teardown reminder | — | — |

Everything above step 3 is free, and step 2 fails loudly if `adapters-both` is
not in the workspace — so a missing artifact costs nothing rather than being
discovered after 40 minutes of a billing A100.

Step 7 is slow because the container downloads ~16 GB of base weights at
startup. That is why the probes in `deployment.yaml` are set to ~35 minutes of
startup budget instead of the default ~5.

### The smoke test

One chat completion per served name, verdicts computed **locally**:

| served name | check |
| --- | --- |
| `meridian` | `data/verifier.py --tenant meridian` on the response text |
| `vantage` | `data/verifier.py --tenant vantage` on the response text |
| `base` | returns text at all — no contract to satisfy |

`base` also gets run through both tenant contracts, printed as **context, not
the verdict**: it is the control arm, and FAIL on both is the correct result.

A one-request-per-arm smoke test is not the separation measurement. That is
`eval/separation.py` against the sealed goal set.

## Why a built image instead of `image: vllm/vllm-openai:v0.27.1`

The cheapest deployment would name the public vLLM image directly — no ACR build
minutes, no push, a faster first deploy. It does not work here, for two
independent reasons:

1. **No way to pass the flags.** The
   [managed online deployment schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online)
   has no `command`, `args` or entrypoint key, and the custom-container how-to
   describes configuring a stock image only through *environment variables its
   own entrypoint reads*. The vLLM image's entrypoint is `vllm serve`, which
   takes the model positionally and every flag on ARGV. There is no env var for
   `--enable-lora` or `--lora-modules`, so a bare image reference cannot serve
   LoRA adapters at all.
2. **No place to measure.** The GPU memory numbers come from `nvidia-smi` run
   inside the container at named phases. That needs a process of ours.

`train/azureml/job_devmatrix.yaml` *does* use the public image with no build —
because a command **job** has a `command:` key. Deployments do not. Same image,
different lever.

The routes are declared in **`environment.yaml`**, not the deployment:
`inference_config` is a key on the environment schema and does not exist on the
deployment schema. The environment may be registered and referenced by version
(what this project does — one build, reused by every redeploy) or inlined in the
deployment YAML; the docs' own CLI sample inlines it.

## The cost guardrail

Before any billable command, `deploy.sh` prints the SKU, the rate, the session
projection for `--hours N` (default 3), and refuses to continue until the
operator types `yes-bill` exactly. In a non-interactive shell it refuses
outright unless `ASSUME_YES=yes-bill` is set.

The rate ($3.673/hour) is a **hardcoded estimate** from the Azure pricing page
and is printed marked as such, with the URL to verify it at run time. It has not
been checked against a real invoice, a region, a currency or any credit.

Quota, checked live 2026-08-25 in southcentralus: `standardNCADSA100v4Family`
32 cores, `TotalDedicatedCores` 52. A managed online deployment reserves an
extra 20%, so this one charges 24 × 1.2 = 28.8 cores against both limits. One
instance fits; a second does not.

## Teardown, and why it verifies

```bash
./serve/azure/deploy.sh teardown
```

Deletes the deployment, then the endpoint, then **lists the endpoints in the
workspace and checks the name is gone**. A delete command that returned zero is
not evidence that the resource is gone; a listing is (change_log entry 10 is
what that lesson cost). If the name is still there, teardown exits nonzero with
a loud block telling you that you are probably still being billed and how to
force it.

Teardown deliberately does **not** touch `adapters-both` or the per-tenant
adapter assets. They cost pennies a month and they are the artifact. To remove
everything including the workspace's storage, key vault and container registry:

```bash
"$AZ_BIN" group delete --name rg-multilora --yes
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
probe https://<ep>.southcentralus.inference.ml.azure.com/score -> HTTP 200
probe https://<ep>.southcentralus.inference.ml.azure.com/v1/chat/completions -> HTTP 404
CHAT COMPLETIONS URL: https://<ep>.southcentralus.inference.ml.azure.com/score
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
    --resource-group rg-multilora --workspace-name mlw-multilora \
    --container inference-server --lines 5000
```

replays it. `./serve/azure/deploy.sh logs` wraps that, saves the full log to
`serve/azure/logs/deployment_logs.txt`, and greps out the phase markers, the
adapter sizes on disk, and the resolved adapter root.

**The number that matters** is the delta between
`before_first_request_<name>` and `after_first_request_<name>`. That is the
adapter's GPU footprint, and it is what replaces the 0.08 GB estimate in
`bench/economics.py`.

For scale: the dev-matrix job (change_log entry 11) measured 0 MiB idle →
75,730 MiB with the server up at vLLM's default 0.92 utilization. That number is
the *server*, not the adapters; it is why the per-adapter delta has to be
bracketed rather than inferred.

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
