# multi-lora

Two LoRA adapters, two fictional tenants, one task. Both tenants turn a leadership-intent
statement into a structured JSON execution plan, but they have opposite house styles.
Meridian Industrial is a regulated manufacturer: formal, gate reviews, compliance notes.
Vantage Cloud is a fast SaaS company: terse, OKRs, sprints, owners. A deterministic
pure-Python verifier (`data/verifier.py`) enforces both contracts. It is the data filter
during generation, the gate on what reaches training, and the measurement at the end.
It has no model client and no network dependency, so the same code produces the same
verdict on any machine.

**Status: complete and measured.** Both adapters are trained, both were served
together from one A100 behind an Azure ML managed online endpoint, and the sealed
evaluation has been run once and reported. The endpoint has been torn down and the
deletion verified. Every stage is in `change_log.md`, 17 entries, mistakes included.

## Results

Full write-up with every table and every citation: **[RESULTS.md](RESULTS.md)**.

Two headline numbers:

- **Sealed separation 100 / 0 / 0.** On 160 goals held out of training and
  hash-locked before the run, the Meridian adapter satisfied its own contract
  160/160 and its rival's 0/160; the Vantage adapter the exact reverse; the base
  model, with the identical system message, satisfied neither on any goal. 480
  requests, 0 errors. Source: `eval/logs/separation_matrix_sealed.json`.
- **Tenant switch +53 ms p50, below jitter.** Alternating adapters every request
  versus repeating one adapter cost +52.9 ms at p50, against p95-p50 spreads of
  198-562 ms on those same arms (`bench/logs/matrix_summary_endpoint_session1.json`).
  Corroborated by fresh-server first-request probes: adapter-vs-base -48 ms and
  +4 ms (`bench/logs/swap_time_summary_20260825T12*.json`). Tenants are warm from
  server start; there is no runtime cold path in this serving design.

The cost side of the ledger is in RESULTS too: multi-LoRA serving measured
**+9.3% TTFT and -10.0% tokens/sec** against a base-only arm, roughly double the
5% that was hoped for.

## Honest limits

`RESULTS.md` section (f), "What breaks", is the list: the verifier checks
vocabulary and schema rather than semantics; the two dev-set vantage misses and
the exact words that caused them; one GPU is shared throughput; the LoRA compute
tax; static adapter registration bounds the resident tenant count and thrashing
beyond it is unmeasured; the evaluation data is synthetic from a single
generator; the NeMo-API training route is dead in the 26.08 container; the load
driver is a stdlib fallback; and every serving number is one day, one region,
list price.

## How it was built

Stage 0 is the verifier, six fixture cases with fixed expected verdicts, the staged data
pipeline, and a 10-goal dry run that exercises the pipeline plumbing on hand-authored
fixtures. Stage 1 adds the measurement tooling — the separation harness, the swap-time
and serving benchmarks, the economics table and the report generator — plus a mock
server that makes all of it self-testable offline. The tooling was built and self-tested
against that mock before any adapter existed, so the instrument was verifiable before
there was a result to attach to it. Stages 2 to 4 are data generation, training, the
live endpoint, and the sealed matrix.

## Layout

```
RESULTS.md             every measured number, with the log file behind each one
ARCHITECTURE.md        how the whole thing works, for a non-specialist reader
change_log.md          the project's memory: 17 entries, mistakes included
data/verifier.py       deterministic schema + vocabulary checker (stdlib only)
data/test_outputs.py   the fixture verdict table, pytest or plain python
data/generate.py       goals -> outputs -> filter -> package
data/fixtures/         six verdict fixtures + the dry-run fixtures
data/logs/             console output for every check in this stage
tools/                 mock vLLM OpenAI server, for running the tools offline
eval/separation.py     the proof artifact: per-arm tenant confusion matrix
bench/                 swap_time.py, run_matrix.py, economics.py
scripts/               guardrail + make_report.py
tests/                 self-tests: measurement tooling, data-generation providers
train/                 two LoRA routes (NeMo primary, HF PEFT fallback) + configs
serve/azure/           vLLM container, az ml YAMLs, deploy.sh with a cost guardrail
serve/spark/           the on-prem DGX Spark path
```

Both stacks have been run: `train/train_lora_hf.py` produced both adapters on an
Azure ML A100, and `serve/azure/deploy.sh` took the endpoint from zero to scoring
URI to teardown. The "Unverified items" section near the bottom of this file
tracks what documentation did not settle; the items the deploy session answered
are marked RESOLVED there.

## Run the verifier self-test

```bash
python data/verifier.py --self-test        # fixture table, exits 0 only on a full match
python data/test_outputs.py                # same verdicts as assertions
python -m pytest data/test_outputs.py -q   # same file under pytest
```

Check individual files:

```bash
python data/verifier.py --tenant meridian data/fixtures/meridian_good.json
python data/verifier.py --help
```

Saved output: `data/logs/verifier_selftest.log`.

## Run the dry run

The dry run proves the pipeline plumbing without calling a model. It reads 20
hand-authored outputs (10 goals x 2 tenants), of which 3 are deliberately broken.

```bash
python data/generate.py filter --input data/fixtures/dryrun_outputs.jsonl \
    --out-dir data/generated/dryrun/
python data/generate.py package --out-dir data/generated/dryrun/ \
    --goals data/fixtures/dryrun_goals.jsonl
```

Expected: 20 in, 17 kept, 3 rejected — one bad gate sequence, one string
`timeline_weeks`, one Vantage plan written in Meridian voice. Saved output:
`data/logs/dryrun.log`.

## Data generation

`data/generate.py outputs` is the only stage that talks to a model, and it can
talk to two:

| flag | transport | key | used for |
| --- | --- | --- | --- |
| `--provider anthropic` (default) | the `anthropic` SDK, imported lazily | `ANTHROPIC_API_KEY` | the original plan |
| `--provider openai` | stdlib `urllib.request`, no new dependency | `OPENAI_API_KEY`, or `--api-key-env NAME` | Azure OpenAI / Azure AI Foundry, and the offline mock |

The second provider exists for a budget reason, not a technical one. This
project's budget is an Azure free-trial credit and there is no Anthropic key in
the build environment, so generation is paid for with Azure credit against a
cheap Azure-hosted deployment rather than blocked indefinitely. The deployment
actually used for all 1,600 rows was `gpt-5-mini` (Azure OpenAI, in this same
subscription, `reasoning_effort: minimal`) — the only current small model this
subscription had deployment quota for; see change_log entry 6. The `gpt-4o-mini`
in the examples below is a placeholder model name.

```bash
# Azure OpenAI, deployment route: POST {base}/openai/deployments/{model}
#   /chat/completions?api-version=VER   with an `api-key` header
.venv/bin/python data/generate.py outputs --provider openai \
    --model gpt-4o-mini \
    --base-url https://my-resource.openai.azure.com \
    --api-key-env AZURE_OPENAI_API_KEY \
    --azure-api-version 2024-10-21

# plain OpenAI route: POST {base}/chat/completions  with Authorization: Bearer
.venv/bin/python data/generate.py outputs --provider openai \
    --model gpt-4o-mini --base-url https://api.openai.com/v1
```

Both providers send the identical request: the same system prompt, one user
message, `max_tokens` 2000, and no sampling parameters unless `--temperature` is
passed. The key is only ever read from an environment variable — there is no
`--api-key` flag — and a missing key exits 2 without sending anything. The one
exception is a loopback `--base-url` (`127.0.0.1`/`localhost`), where a missing
key is allowed because that is the mock server and it has no auth. Failures are
retried up to 3 attempts on HTTP 429/5xx with fixed 2s/4s sleeps between them,
after which the row is recorded with an `error` field and the run continues; the
stage summary counts them.

**Which generator produced a row is not load-bearing.** Every row goes through
`data/verifier.py` in the filter stage, which is deterministic, has no model
client and no network. A cheaper or different generator changes cost, latency
and rejection rate — never what is allowed into training. That is the whole
reason switching to an Azure-hosted model is a budget decision rather than a
scientific one.

### Self-test offline

```bash
python -m pytest tests/test_generate_openai_provider.py -q   # 5 tests
python tests/test_generate_openai_provider.py                # same, plain runner
```

It starts `tools/mock_openai_server.py`, runs the `outputs` stage against it
with `--provider openai` over 3 goals from `data/fixtures/dryrun_goals.jsonl`,
and asserts 6 rows (3 goals x 2 tenants) with non-empty text and 0 errors — once
with a key present and once with none, on loopback. It then pipes those rows
through the filter stage: each tenant's mock plan passes its own contract
(6 kept, 0 rejected) and the Meridian plan filed under Vantage is rejected 3/3,
so "everything passed" cannot be the filter waving data through. No key, no
network beyond loopback, no credit spent. Saved output:
`data/logs/generate_provider_selftest.log`.

## Guardrail

```bash
bash scripts/check_no_model_imports.sh     # exits nonzero if a model client appears
```

Saved output: `data/logs/guardrail_grep.log`.

## Measurement tooling

The measurement is built before the thing it measures. Every tool below runs
today, offline, against `tools/mock_openai_server.py` — a stdlib mock of a vLLM
OpenAI-compatible server with per-request LoRA selection, configurable TTFT and
inter-token delay, and a configurable extra delay on the first request to each
adapter that stands in for a cold adapter load. The same tools then point at the
real Azure scoring URI by changing `--endpoint`, and that is exactly what
happened at Stages 3 and 4 — the published numbers came out of these tools
unmodified.

| tool | what it measures |
| --- | --- |
| `tools/mock_openai_server.py` | nothing — it is the offline stand-in being measured against |
| `eval/separation.py` | per-arm confusion matrix: how often base / meridian / vantage output passes each tenant's contract |
| `bench/swap_time.py` | cold vs warm adapter swap, isolated by controlling request order |
| `bench/run_matrix.py` | four-arm serving benchmark: arm and metric contract plus a fallback load driver |
| `bench/economics.py` | GPU memory and cost per tenant per month for N tenants (arithmetic, no network) |
| `scripts/make_report.py` | renders `reports/iter_NN.md` from raw logs, citing the source file for every number |

Two rules hold across all of them. Nothing under `eval/` or `bench/` imports a
model-client library; HTTP is `urllib.request` from the stdlib. And every script
writes raw per-request JSONL first and computes its summary by reading that file
back off disk, so each summary carries the path of the log it came from.

### Self-test offline

```bash
python -m pytest tests/test_tooling.py -q   # 8 tests, starts the mock itself
python tests/test_tooling.py                # same tests, plain runner
```

The mock returns a fixed valid Meridian plan for model `meridian`, a fixed valid
Vantage plan for `vantage`, and non-JSON prose for `base`, so the correct
confusion matrix is known in advance (100/0, 0/100, 0/0) and the harness is wrong
if it reports anything else. The same trick fixes the cold-swap answer: the mock
injects a known delay and `swap_time.py` has to recover it.

Saved output: `data/logs/tooling_selftest.log`.

### Synthetic-number warning

`reports/iter_00.md` and every log file ending in `_selftest` were produced
against the mock server. **Those numbers describe the measuring instrument, not
the system under test**, and `iter_00.md` says so on its first line. Do not quote
them. The real numbers are in `reports/iter_01.md` through `iter_04.md` and
collected in `RESULTS.md`.

The adapter size of 0.08 GB that `bench/economics.py` still carries as its
default is likewise an estimate. It has been superseded: the measured artifact is
167,832,240 bytes (~0.168 GB), and `bench/logs/economics_measured_final.md` is
the table produced with `--adapter-gb 0.168`.

### Run a tool by hand

```bash
python tools/mock_openai_server.py --port 8000 --ttft-ms 80 --itl-ms 10 \
    --cold-first-request-ms 400 &

python eval/separation.py --endpoint http://127.0.0.1:8000 \
    --api-key-env MULTILORA_API_KEY --goals tests/fixtures/tooling_goals.jsonl
python bench/swap_time.py --endpoint http://127.0.0.1:8000 --adapter meridian
python bench/run_matrix.py --endpoint http://127.0.0.1:8000 --requests-per-arm 24
python bench/economics.py
python scripts/make_report.py --iter 1 --objective "..."
```

The bearer key is only ever read from an environment variable named by
`--api-key-env`; no tool accepts a key as a flag.

### Sealed set

The headline separation number has to come from a goal set fixed before it was
ever run against. `eval/separation.py --make-sealed-hash FILE` writes
`eval/SEALED.sha256` once and refuses to overwrite it; `--sealed` then refuses to
run unless the goals file still matches that hash, and refuses a second sealed
run unless `--allow-rerun` is given. The sealed set is `eval/sealed_goals.jsonl`
(goals 801-960, which never had training outputs generated at all), hashed into
`eval/SEALED.sha256` at Stage 1 and run exactly once, at Stage 4. The hash is
`d54319eb…47fb15` and it was checked before the first request went out
(`data/logs/sealed_run.log`, first lines).

## Training

Two routes to the same artifact: a rank-16 LoRA adapter per tenant in Hugging
Face PEFT layout. Both read the same `train/config_<tenant>.yaml`, so the
hyperparameters cannot drift between them. Full detail in `train/README.md`.

| | route A (primary) | route B (fallback) |
| --- | --- | --- |
| script | `train/train_lora.py` | `train/train_lora_hf.py` |
| stack | NeMo Framework 2.x | HF PEFT + TRL `SFTTrainer` |
| environment | `nvcr.io/nvidia/nemo` container | `pip install torch transformers peft trl datasets` |
| output | `.nemo`, then `train/convert_to_hf.py` | HF PEFT adapter directly |

```bash
# resolves the config, converts the data, prints the API calls, imports nothing
.venv/bin/python train/train_lora.py    --tenant meridian --dry-run
.venv/bin/python train/train_lora_hf.py --tenant meridian --dry-run

# the real thing, on an A100
python train/train_lora.py --tenant meridian          # route A
python train/convert_to_hf.py --tenant meridian       # .nemo -> HF PEFT
python train/train_lora_hf.py --tenant meridian       # route B, no conversion

# either route, before serving
python train/convert_to_hf.py --verify train/out/meridian_hf
```

Hyperparameters: rank 16, alpha 32, dropout 0.05, seven target projections,
seed 1234, bf16, 3 epochs, lr 1e-4 cosine, global batch 8, seq len 2048.

Two things worth knowing before reading the code. **NeMo calls the LoRA rank
`dim`; PEFT calls it `r`** — same number, two spellings, and the likeliest place
for the two routes to diverge. And **NeMo does not accept Hugging Face module
names**: Megatron fuses q/k/v into `linear_qkv` and gate/up into `linear_fc1`,
so `train_lora.py` translates seven HF names into four NeMo ones and prints the
translation before it runs.

A route switch goes through `change_log.md`, not a commit message. Adapters
trained by different routes are not interchangeable evidence.

**What actually ran: route B.** The `nvcr.io/nvidia/nemo:26.08` container's
default interpreter ships `megatron.bridge` but neither `nemo` nor `lightning`,
so route A failed on import and the delivered adapters were trained with HF PEFT
+ TRL inside that same container (`change_log.md` entry 9). Both adapters are
rank-16, ~13 minutes each on one A100, artifact 167,832,240 bytes each.

## Serving

One frozen 8B base in GPU memory, both adapters resident beside it, adapter
selection per request via the OpenAI `model` field (`base`, `meridian`,
`vantage`). `ARCHITECTURE.md` explains why the swap is cheap; `serve/azure/README.md`
is the operational runbook.

**Primary — Azure ML managed online endpoint**, custom vLLM container, 1x A100
(`Standard_NC24ads_A100_v4`). What ran: endpoint `multilora-ep`, deployment
`blue`, region `southcentralus`, image `vllm/vllm-openai:v0.27.1`, on 2026-08-25.
Torn down; deletion read back (`serve/azure/logs/teardown_session1.log`).

```bash
./serve/azure/deploy.sh cost        # see the bill before agreeing to it
./serve/azure/deploy.sh --hours 3   # zero -> scoring URI -> smoke test
./serve/azure/deploy.sh teardown    # MANDATORY. stops the billing, then verifies it
```

`deploy.sh` refuses to create anything billable until the operator types
`yes-bill`, and prints the SKU, the hourly rate and the projected session cost
first. `teardown` deletes the deployment and endpoint and then **lists the
endpoints to confirm the name is gone** — a delete that returned zero is not
evidence. An idle A100 endpoint bills about ₹320/hour whether or not a request
ever arrives.

**Secondary — DGX Spark**, on-prem: the same `vllm serve` configuration, but NOT
the same image — Spark is arm64 + GB10 Blackwell, so the launch script targets
NVIDIA's dedicated Spark container path (see the GB10 rows in "Unverified
items"); untested on GB10 hardware:

```bash
./serve/spark/launch.sh --check-image   # resolve the GB10/sm_121 question on the box
./serve/spark/launch.sh
```

Numbers from the Spark are hardware-dependent (GB10, unified 128 GB memory pool)
and are not comparable with the A100 numbers. The script says so and labels its
output accordingly.

GPU memory is captured by having `start_server.sh` print timestamped
`nvidia-smi` samples to stdout at named phases, then recovering them with
`az ml online-deployment get-logs`. The intent was that the delta across
`before_first_request_<name>` / `after_first_request_<name>` would be the
adapter's GPU footprint. **It is not, and the instrument says so:** every phase
from `server_ready_base_resident` onward reads a flat 74,002 MiB, because vLLM
pre-allocates its pool at `--gpu-memory-utilization 0.90` before any adapter is
touched. Per-adapter GPU footprint is therefore not measurable this way. The
honest per-tenant number is the artifact size, 0.168 GB. Full phase table in
`RESULTS.md` section (d).

## Reproduce

From a clean clone. Steps 1 and 2 are free and offline. Steps 3 onward spend
money — 4, 5 and 6 run on an A100 and **bill by wall-clock time**.

```bash
# 1. prove the instrument, offline, no key, no GPU, no network beyond loopback
python -m venv .venv && .venv/bin/pip install pyyaml
.venv/bin/python data/verifier.py --self-test          # 13/13 fixture verdicts
.venv/bin/python -m pytest tests/ data/test_outputs.py -q
bash scripts/check_no_model_imports.sh
.venv/bin/python scripts/check_flag_continuation.py    # entry-10 lesson, executable

# 2. pipeline plumbing on fixtures, still no model
.venv/bin/python data/generate.py filter --input data/fixtures/dryrun_outputs.jsonl \
    --out-dir data/generated/dryrun/

# 3. data. costs money: ~1,600 calls to an Azure OpenAI deployment
.venv/bin/python data/generate.py goals   --out-dir data/generated/full/ --n-goals 960 --seed 1234
.venv/bin/python data/generate.py outputs --provider openai --model "$GEN_MODEL" \
    --base-url "$AZURE_OPENAI_ENDPOINT" --api-key-env AZURE_OPENAI_API_KEY \
    --azure-api-version "$AZURE_OPENAI_API_VERSION" \
    --max-tokens-param max_completion_tokens --reasoning-effort minimal --concurrency 8
.venv/bin/python data/generate.py filter  --out-dir data/generated/full/
.venv/bin/python data/generate.py package --out-dir data/generated/full/
.venv/bin/python eval/separation.py --make-sealed-hash eval/sealed_goals.jsonl

# 4. train, one job per tenant, ~13 min each on one A100
az ml job create -f train/azureml/job_hf.yaml -g "$RESOURCE_GROUP" -w "$WORKSPACE" \
    --set inputs.tenant=meridian --set display_name=hf-lora-meridian
az ml job create -f train/azureml/job_hf.yaml -g "$RESOURCE_GROUP" -w "$WORKSPACE" \
    --set inputs.tenant=vantage  --set display_name=hf-lora-vantage
az ml job create -f train/azureml/job_combine_adapters.yaml -g "$RESOURCE_GROUP" -w "$WORKSPACE"
# then register the job output as the `adapters-both` custom_model asset
# (exact command in the header of job_combine_adapters.yaml)

# 5. serve. THIS IS THE BILLABLE ONE. ~46 min to a working scoring URI.
./serve/azure/deploy.sh cost          # prints SKU, rate, projected session cost
./serve/azure/deploy.sh --hours 3     # refuses to proceed until you type yes-bill
source serve/azure/logs/endpoint.env  # MULTILORA_CHAT_URL + MULTILORA_API_KEY

# 6. measure
.venv/bin/python bench/run_matrix.py --endpoint "$MULTILORA_CHAT_URL" \
    --api-key-env MULTILORA_API_KEY --requests-per-arm 40 --concurrency 4
.venv/bin/python bench/swap_time.py  --endpoint "$MULTILORA_CHAT_URL" \
    --api-key-env MULTILORA_API_KEY --adapter meridian
.venv/bin/python eval/separation.py  --endpoint "$MULTILORA_CHAT_URL" \
    --api-key-env MULTILORA_API_KEY --goals eval/sealed_goals.jsonl --sealed
.venv/bin/python bench/economics.py --adapter-gb 0.168 --sku-price-usd-hr 3.673 \
    --hours-month 730

# 7. MANDATORY
./serve/azure/deploy.sh teardown      # deletes, then reads the endpoint list back
```

Two notes on running this somewhere else.

**Every Azure resource name is an environment variable with a default, not a
hardcoded string.** `deploy.sh` reads `SUBSCRIPTION_ID`, `LOCATION`,
`RESOURCE_GROUP`, `WORKSPACE`, `ENDPOINT_NAME`, `DEPLOYMENT_NAME`,
`ENVIRONMENT_NAME`, `ADAPTER_MODEL_NAME`, `INSTANCE_TYPE`, `SKU_USD_PER_HOUR` and
`AZ_BIN`, each as `${VAR:-default}`. Export them and the same script runs in
another subscription, region or workspace with no code edit. The served adapter
names are `--served-names` on every measurement tool for the same reason.

**Steps 4, 5 and 6 bill for GPU time whether or not a request arrives.** Deploy
step 5 last, run step 6 immediately, run step 7 the moment you are done. An idle
A100 endpoint costs the same as a busy one. `change_log.md` entry 10 is what it
looks like when that goes wrong: 3.2 hours of A100, zero requests served, about
$11.70.

## Unverified items (# CHECK list)

Every item below is a flag, key, class name or behaviour that official
documentation did **not** settle. Each is marked `# CHECK:` at the exact line in
the file. An honest gap beats a confident guess: a plausible-looking invented
flag fails at the most expensive possible moment.

Doc research date for all of it: **2026-08-24**. Items the training and deploy
sessions of 2026-08-25 answered are marked **RESOLVED** with the answer and the
log that carries it. The rest are still open.

### NeMo / training

| file:line | item |
| --- | --- |
| `train/config_meridian.yaml:28`, `train/config_vantage.yaml:28` | NeMo config/model class names for `Llama-3.1-Nemotron-Nano-8B-v1`. Docs publish an `import_ckpt` example only for the Ultra 253B variant. `Llama31NemotronNano8BConfig` is a guess. |
| `train/config_meridian.yaml:120`, `train/config_vantage.yaml:120` | `ChatDataModule` — described only as "sets a few default arguments on top of `FineTuningDataModule`". Import path and expected JSONL schema unpublished. If it takes `{"messages": [...]}` directly, the conversion in `train_lora.py` should be deleted. |
| `train/train_lora.py:14` | **RESOLVED (badly) 2026-08-25** — the image used was `nvcr.io/nvidia/nemo:26.08`, and its default interpreter (`/opt/venv/bin/python` 3.12.3) ships `megatron.bridge` but neither `nemo` nor `lightning`. Route A fails on `import nemo`. NVIDIA moved the 26.x training stack to Megatron-Bridge; NeMo source trees sit uninstalled in `/opt`. Evidence: jobs `upbeat_map_g80sxygzbh` (failed) and `funny_ball_xpnt0wxsxt` (module table), `train/logs/container_diagnostic_20260825.txt`, `change_log.md` entry 9. The two adapters were trained by route B. Every route-A row above is therefore untested against a real NeMo import and stays open. |
| `train/train_lora.py:191` | How `FineTuningDataModule` wraps `input` in the model's chat template. If training and serving disagree on prompt shape, the separation number is quietly low with no error. |
| `train/train_lora.py:354`, `train/train_lora.py:449` | Same class-name gap as above, at the call site. The script refuses to run on an unresolvable name rather than substituting one. |
| `train/train_lora.py:429` | Seed pinning. NeMo 2.x documents no seed argument on `llm.finetune` or `nl.Trainer`; `seed_everything` is used, but whether it reaches Megatron's data sampler and parallel RNG is unconfirmed. Treat same-seed runs as reproducible-ish, not bit-identical. |
| `train/convert_to_hf.py:36` | Whether a supported command-line NeMo exporter exists. Docs show the Python API (`llm.export_ckpt(target='hf-peft')`) only. |
| `train/train_lora_hf.py:317` | `assistant_only_loss` needs a chat template containing `{% generation %}` markers, auto-patched by TRL only "for known model families (e.g. Qwen3)". Whether the Nemotron Nano template has them is unverified, so the flag is off by default. |

### Azure ML / serving

| file:line | item |
| --- | --- |
| `serve/azure/environment.yaml:79` | **RESOLVED 2026-08-25** — Azure honoured `scoring_route.path`, so the scoring URI *is* `https://multilora-ep.southcentralus.inference.ml.azure.com/v1/chat/completions`. It answered HTTP 200; appending a second `/v1/chat/completions` returned 424. Every tool worked unmodified and no rewrite proxy was needed. Evidence: `serve/azure/logs/deploy_session1.log`, the "probe which URL actually serves chat completions" block. |
| `serve/azure/deployment.yaml:71` | **RESOLVED 2026-08-25** — nested. The adapters landed at `/mnt/adapters/adapters/<tenant>`, i.e. the folder-shaped `custom_model` keeps its own top-level folder name inside the mount, and `AZUREML_MODEL_DIR` did follow the custom `model_mount_path`. `start_server.sh` logged `adapter root /mnt/adapters/adapters` and `matched pattern <root>/*/meridian/adapter_config.json`. Evidence: `serve/azure/logs/deployment_logs.txt`, first lines. |
| `serve/azure/deployment.yaml:210` | **RESOLVED for this case 2026-08-25** — no provisioning timeout was hit. Deployment `blue` created 11:13:44Z and took 100% of traffic at 11:56:36Z: 42.9 minutes, inside the 20-40 minute expectation the script prints and inside the probes' 900s `initial_delay` plus 40 retries. Inside the container it was much faster — process start 11:39:50Z to `/health` answering at 11:42:16Z, 2 min 26 s, base weights and both adapters included. Azure's own limit remains undocumented; what is known is that this deployment did not hit one. Evidence: `serve/azure/logs/deploy_session1.log`, `serve/azure/logs/deployment_logs.txt`. |
| — | **RESOLVED 2026-08-25: the ACR build.** Never run before; it worked. The BYOC environment `multilora-vllm:1` registered in 31s (11:11:26Z -> 11:11:57Z) and the workspace registry built the image with no manual intervention. Whole `deploy.sh` run, pre-flight to smoke test complete: 11:11:20Z -> 11:57:18Z, 46 minutes. Evidence: `serve/azure/logs/deploy_session1.log`. |
| — | **RESOLVED 2026-08-25: adapters are preloaded, not lazily loaded.** vLLM loads everything named in `--lora-modules` during server startup, so both tenants were GPU-resident before the first client request. This is why `bench/swap_time.py` found no runtime cold path. Consequence, now a limit rather than a gap: the resident tenant set is fixed at launch and bounded by `--max-loras` / `--max-cpu-loras`. Evidence: `bench/logs/swap_time_summary_20260825T12*.json`, `change_log.md` entry 15. |
| — | **STILL OPEN: per-adapter GPU footprint.** The phase markers cannot see it through vLLM's pre-allocated pool — flat 74,002 MiB from server-ready onward. Measuring it needs an instrument inside the vLLM process, not `nvidia-smi`. `RESULTS.md` section (d). |
| — | **STILL OPEN: behaviour beyond `--max-loras`.** Two tenants, four slots. Eviction, reload and thrash cost are unmeasured. |

### DGX Spark

| file:line | item |
| --- | --- |
| `serve/spark/launch.sh:51` | Whether any **pinned, reproducible** vLLM tag supports GB10/`sm_121`. `vllm/vllm-openai` does publish aarch64 tags (`v0.27.1-aarch64`, verified on Docker Hub), but those are CUDA 12.9 while GB10 wants CUDA 13. The official vLLM DGX Spark post recommends `cu130-nightly` and warns it is "a compatibility track rather than a reproducible pin". |
| `serve/spark/launch.sh:60`, `:80` | `DEFAULT_IMAGE` is therefore a moving nightly. Override `IMAGE` with a pinned digest for any published number. `--check-image` resolves this on the box: it pulls the tag, prints the digest, the architecture, the CUDA/torch build and the `sm_` capability vLLM sees. |

### Data generation / Azure OpenAI

Doc research date: **2026-08-25**. The request shape itself *was* verified —
`POST https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=YYYY-MM-DD`
with the key in an `api-key` header, and the newer v1 route
(`{endpoint}/openai/v1/`, no `api-version`, deployment name in the body's
`model` field) as OpenAI-client compatible
([reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference),
[v1 API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)).
What the docs could not settle:

| file:line | item |
| --- | --- |
| `data/generate.py:381` | **RESOLVED 2026-08-25** — the deployment was named after the model (`gpt-5-mini`), so reusing `--model` for both worked and no `DeploymentNotFound` was seen across 1,600 calls. The gap itself is real and stays a gap for any resource that names deployments differently. Evidence: `data/logs/pilot_outputs.log`, `data/logs/full_outputs.log`. |
| `data/generate.py:385` | **RESOLVED 2026-08-25** — the resource accepted `api-version 2025-04-01-preview`. Passed as `--azure-api-version`, still never hardcoded, and still resource-specific. Evidence: `data/logs/pilot_outputs.log`. |
| `data/generate.py:388` | Azure AI Foundry models that are **not** Azure OpenAI (serverless Foundry Models — Mistral, DeepSeek, Llama) are documented on a different route: `POST /chat/completions?api-version=...` under a `/models` base with `Authorization: Bearer`, not `/openai/deployments/`. The exact base path was not confirmed. For those, omit `--azure-api-version` and put the full base path in `--base-url`. |
| `data/generate.py:409` | `api-key` vs `Authorization: Bearer`. `api-key` is the documented header for Azure OpenAI **key** auth and `Bearer` is documented for Entra ID tokens, but the docs' own OpenAI-client examples for the v1 route pass a key that the client sends as `Bearer`. The two are therefore not interchangeable across the two routes; this code pairs `api-key` with the deployment route, which is the documented pairing. |
| — | RESOLVED 2026-08-25: `--max-tokens-param max_completion_tokens` selects the field name for reasoning models, and `--reasoning-effort` sends `reasoning_effort` when set. Confirmed against the live `gpt-5-mini` deployment (smoke test, `data/logs/pilot_outputs.log`): the deployment route with `api-version 2025-04-01-preview` accepted `max_completion_tokens` + `reasoning_effort: minimal`. |

### Carried over from earlier stages

| file:line | item |
| --- | --- |
| `data/generate.py:55` | **STILL OPEN.** Default generator model id `claude-sonnet-5`. Never exercised — every generated row in this repo came from `--provider openai` against Azure OpenAI `gpt-5-mini`. |
| `bench/run_matrix.py:171`, `:179`, `:183` | **STILL OPEN, and it did not get swapped.** The Stage 3 benchmark ran on the stdlib fallback driver. genai-perf and vLLM's `benchmark_serving.py` remain the preferred generators, and one of them gives a true per-token-pair ITL distribution the fallback cannot. The arms and the 17-field metric row are final; the driver is not. Every number in `RESULTS.md` section (b) carries this caveat. |

### What *was* verified

Listed so the CHECK list above is read as the exception, not the rule. Every
one of these was confirmed against the official documentation on 2026-08-24:

- **vLLM flags** — `--enable-lora`, `--max-lora-rank` (default 16), `--max-loras`
  ("Max number of LoRAs in a single batch", default 1), `--max-cpu-loras`
  ("Must be >= than `max_loras`"), `--served-model-name`, `--gpu-memory-utilization`
  (default 0.92), `--max-model-len`, `--download-dir`, `--port` (default 8000),
  `--host`, `--api-key`, `--max-num-seqs`
  ([engine args](https://docs.vllm.ai/en/latest/configuration/engine_args.html),
  [serve CLI](https://docs.vllm.ai/en/latest/cli/serve.html));
  `--lora-modules` in both `name=path` and JSON forms, adapter selection via the
  request's `model` field, and the runtime LoRA API
  ([LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html)).
- **vLLM `/health`** as a documented endpoint, used for both probes
  ([online serving](https://docs.vllm.ai/en/latest/serving/online_serving/)).
- **`vllm serve` takes the model POSITIONALLY.** `--model` is rejected with
  "you should provide the model as a positional argument", and is slated for
  removal ([PR 16691](https://github.com/vllm-project/vllm/pull/16691)). This is
  a deviation from the task spec, and a deliberate one.
- **vLLM image** `vllm/vllm-openai:v0.27.1`, pushed 2026-08-11
  ([Docker Hub tags](https://hub.docker.com/r/vllm/vllm-openai/tags)); upstream
  `ENTRYPOINT ["vllm", "serve"]`
  ([Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile)).
- **Azure ML endpoint YAML** — `$schema`, `name` (required), `auth_mode`
  (`key` | `aml_token` | `aad_token`)
  ([reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-endpoint-online)).
- **Azure ML deployment YAML** — `endpoint_name`, `model`, `model_mount_path`,
  `environment`, `instance_type`, `instance_count`, `environment_variables`,
  `request_settings.request_timeout_ms` (**max 180000 ms**, default 5000),
  `max_concurrent_requests_per_instance` (default 1), `liveness_probe` /
  `readiness_probe` (`initial_delay`, `period`, `timeout`, `success_threshold`,
  `failure_threshold` — and **no** `path`/`port` keys),
  `egress_public_network_access`, `app_insights_enabled`
  ([reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online)).
- **Azure ML environment YAML** — `image` / `build.path` / `build.dockerfile_path`,
  `os_type`, and `inference_config.{liveness,readiness,scoring}_route.{path,port}`
  ([reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-environment)),
  with `inference_config` required for BYOC
  ([custom container how-to](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-container)).
- **NeMo 2.x** — `llm.peft.LoRA(dim=, alpha=, dropout=, target_modules=)` with
  rank spelled `dim`; NeMo module names `linear_qkv` / `linear_proj` /
  `linear_fc1` / `linear_fc2`; `CanonicalLoRA` as the HF-equivalent form;
  `FineTuningDataModule` expecting `{"input", "output"}` rows in a
  `training.jsonl` / `validation.jsonl` / `test.jsonl` root;
  `MegatronMixedPrecision(precision="bf16-mixed")`; `MegatronOptimizerModule` +
  `CosineAnnealingScheduler`; `llm.export_ckpt(target='hf-peft')`; and that
  NeMo 2.0 replaced YAML config with Python
  ([PEFT guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/sft_peft/peft_nemo2.html)).
- **TRL / PEFT** — `SFTTrainer(model=, args=, train_dataset=, processing_class=,
  peft_config=)`; `SFTConfig` fields including `max_length` (**not**
  `max_seq_length`), `lr_scheduler_type`, `assistant_only_loss`; conversational
  `{"messages": [...]}` datasets get the chat template applied automatically
  ([TRL](https://huggingface.co/docs/trl/en/sft_trainer)); `LoraConfig(r=,
  lora_alpha=, lora_dropout=, target_modules=, bias=, task_type="CAUSAL_LM")`
  writing `adapter_config.json` + `adapter_model.safetensors`
  ([PEFT](https://huggingface.co/docs/peft/en/package_reference/lora)).
- **NIM** — `NIM_PEFT_SOURCE`, the `<dir>/<adapter>/adapter_config.json` layout,
  `NIM_PEFT_REFRESH_INTERVAL`, and that NIM passes `--enable-lora`, `--max-loras`,
  `--max-cpu-loras`, `--max-lora-rank` through to vLLM
  ([NIM LoRA](https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/finetune-lora.html)).

## Finding from Stage 0: schema alone is not enough

`data/fixtures/meridian_crossover.json` passes every Meridian schema check and is still
wrong. Its prose is written in Vantage voice — "owners ship in weeks", "iterate on
blockers". A schema-only filter would have accepted it into training data and taught the
Meridian adapter the wrong house style. The vocabulary check is what catches it, and it
is the reason `ok` requires both `schema_ok` and `vocab_ok`. `vantage_crossover.json` is
the same failure in the other direction.

Two matching behaviours are deliberate. Vocabulary matching runs on the raw output text,
keys included, with `\b` word boundaries. Underscore is a word character, so `\bowner\b`
does not match the key `owner_role` and Meridian's own schema never trips Meridian's
forbidden-term list. A quote is not a word character, so the bare key `"gate"` does match
`\bgates?\b`, which means a Meridian plan auto-fails Vantage rules on its schema alone.
Both are tested.

## Assumptions

- Every number in this repo traces to a log file under `data/logs/`, `bench/logs/`,
  `eval/logs/` or `serve/azure/logs/`. Anything synthetic or fixture-based is
  labelled as such where it appears, including a `_note` marker on the first line
  of both dry-run files. `RESULTS.md` cites the file behind every number it prints.
- `ANTHROPIC_API_KEY` is not present in the build environment, so the default
  provider was never used. All 1,600 generated rows came from `--provider openai`
  against an Azure OpenAI `gpt-5-mini` deployment, funded by the Azure free-trial
  credit. Full-set rejection was 4.69%
  (`data/generated/full/filter_summary.json`). The Stage 0 dry run validates
  pipeline plumbing on labelled fixtures only, and no fixture in this repo is
  training data.
- The local machine has no usable GPU (GTX 1650, 4GB). All training and serving ran
  on Azure, on one `Standard_NC24ads_A100_v4` in `southcentralus`.
- Base model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. The system message
  `detailed thinking off` goes in every training row and every eval call.
- Serving was vLLM with `--enable-lora` behind an Azure ML managed endpoint,
  OpenAI-compatible, bearer key in an `Authorization` header. Adapter selection is
  per-request via the `model` field: `base`, `meridian`, `vantage`. Every tool takes
  `--served-names` so those strings can change without a code edit.
- `bench/swap_time.py` is the only hand-rolled timing tool, and deliberately so:
  isolating a first-touch request needs control over request order that a load
  generator does not give you. It found that this design has no runtime cold path
  at all — adapters are preloaded at server start — which is a stronger result than
  the one it was built to measure. `bench/run_matrix.py` fixes the arms and the
  metric row; the handoff to genai-perf or vLLM's `benchmark_serving.py` did not
  happen and the published benchmark ran on the stdlib fallback.
- `blockers[]` may be empty. The spec is silent on a minimum, so the verifier requires
  the key to be present and to be a list of non-empty strings, and accepts an empty list.
- Unknown top-level keys are rejected for both tenants. Strictness was chosen over
  tolerance because the verifier has to give the same answer every time it runs.
