# Architecture

This document explains how two customers ("tenants") get their own custom model
behaviour out of one GPU. It is written for a reader who does not work with
machine learning day to day. There is no marketing here; where a number is an
estimate rather than a measurement, it says so.

**Measured on.** Everything in this document that is now a measurement rather
than an estimate was measured on one NVIDIA A100 80GB — Azure ML managed online
endpoint, SKU `Standard_NC24ads_A100_v4`, region `southcentralus` — on
2026-08-25, serving `vllm/vllm-openai:v0.27.1`. The results and the log file
behind each number are in `RESULTS.md`. Numbers still carrying a `~` are still
estimates.

Two tenants exist in this project: **Meridian Industrial** (formal, regulated,
gate reviews) and **Vantage Cloud** (terse, sprints, owners). Both take the same
input — a leadership goal in plain English — and return a structured plan. They
differ only in house style and output schema.

## 1. One base model, held once

The base model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. It has about 8 billion
parameters. Stored at bf16 precision (2 bytes per parameter) that is roughly
16 GB of weights.

The base model is loaded into GPU memory once, when the server starts, and it is
**frozen**. Nothing about it changes while the server runs. No tenant can modify
it. It is the same 16 GB no matter how many tenants are being served.

The GPU is one NVIDIA A100 with 80 GB of memory. After the 16 GB of weights, the
rest is working space: the KV cache (the server's short-term memory for
in-flight conversations), activations, and the adapter cache described below.

## 2. What a LoRA adapter actually is

Fine-tuning a model normally means changing all 8 billion of its numbers. That
produces a second 16 GB model. Ten tenants would mean ten 16 GB models.

LoRA (Low-Rank Adaptation) does something cheaper. It leaves the original
weights alone and attaches a small correction beside each targeted layer. The
correction is stored as **two thin matrices**, A and B. Where the original layer
is a big square of numbers, A and B are narrow strips. Their product has the same
shape as the original layer, so it can be added to the layer's output, but it
takes far fewer numbers to write down. How narrow the strips are is the **rank**.
This project uses rank 16.

Seven layer types per transformer block are targeted:

| module | what it is |
| --- | --- |
| `q_proj`, `k_proj`, `v_proj` | the three attention projections |
| `o_proj` | the attention output projection |
| `gate_proj`, `up_proj`, `down_proj` | the three feed-forward projections |

**Size, measured.** The trained adapter is **167,832,240 bytes — 160 MB — and
both tenants' adapters are that size to the byte**, which is expected since the
shape is fixed by the configuration and not by the data. That is fp32 storage of
about 42 million adapter parameters; written at bf16 it would be roughly half.
Source: `change_log.md` entry 11, from the training job output.

> The estimate this replaces was ~84 MB, arithmetic over the published
> Llama-3.1-8B layer dimensions at bf16. The arithmetic was right and the
> storage precision was the thing not known: 42M parameters at 4 bytes is
> 160 MB, at 2 bytes it is 80 MB. `bench/economics.py` still defaults to the old
> 0.08 GB estimate; `bench/logs/economics_measured_final.md` is the table run
> with the measured `--adapter-gb 0.168`.

The comparison that matters: **16,000 MB of base model, 160 MB of tenant.** The
tenant-specific part is about one percent of the whole.

## 3. Why switching tenants is nearly free

The important claim in this project is that moving from Meridian to Vantage
between two requests costs almost nothing. Here is why.

**Warm swap — the normal case.** vLLM is started with `--enable-lora` and a
`--max-loras N` setting. `--max-loras` is documented as "Max number of LoRAs in a
single batch". The adapters occupy GPU-resident slots, and the adapter maths is
applied **per request inside the batched kernel**: the server can have a Meridian
request and a Vantage request in the same batch, each getting its own adapter
applied to the shared base weights. Switching tenants between requests therefore
does not move any model weights. It changes which small matrix the kernel reads
for that row of the batch. In effect it is a pointer change. This is why the
warm swap cost should be near zero, and `bench/swap_time.py` exists to prove that
against the real server rather than assert it.

**Measured:** adapter TTFT p50 minus base TTFT p50 was −48 ms for meridian and
+4 ms for vantage, against a 292 ms p95−p50 spread on the same run. Zero within
noise, in both directions. `RESULTS.md` section (c).

**Cold swap — and why it did not happen.** The first request for an adapter the
server has not served yet would have to fetch that adapter's weights and put them
on the GPU: Azure Blob Storage → the container's local disk → host memory → GPU
memory. On this deployment no request ever pays it. **Adapters named in
`--lora-modules` are loaded during server startup, so every registered tenant is
GPU-resident before the first client request arrives** — meridian's first-ever
request was *faster* than its own warm p50 (0.976s vs 1.232s), and vantage's was
100 ms slower, inside its own warm p95 spread. The Blob-to-GPU cost is real and
is paid once, inside the container start: 2 min 26 s from process start to
`/health` answering, base weights and both adapters included. A per-request cold
number exists only under dynamic adapter loading, which is a different serving
mode; not measured. `--max-cpu-loras` sets how many adapters are kept in host
memory, which would shorten a re-load if an adapter were evicted from the GPU
slots and needed again.

`bench/swap_time.py` measures both by controlling request order: warm baseline
first, then exactly one first-touch request to the adapter (the cold sample),
then repeated requests to the same adapter (the warm samples). Cold minus warm is
the adapter load cost.

## 4. The economics of many tenants

At the measured adapter size of 0.168 GB (`bench/logs/economics_measured_final.md`):

| tenants | separate fine-tuned models | one base + N adapters |
| ---: | ---: | ---: |
| 1 | ~16 GB | 16.17 GB |
| 2 | ~32 GB | 16.34 GB |
| 5 | ~80 GB | 16.84 GB |
| 20 | ~320 GB | 19.36 GB |

An A100 has 80 GB. The left-hand column runs out of GPU at five tenants and needs
a second GPU. The right-hand column has not meaningfully moved. The practical
statement is: **N tenants is one base model plus N small files, not N GPUs.**

The cost consequence is the same shape. One A100 endpoint costs the same per hour
whether it serves one tenant or twenty, so the cost per tenant per month falls
roughly as 1/N. `bench/economics.py` computes this as explicit arithmetic with
named, overridable constants, so the assumptions are visible rather than
asserted.

The honest caveat: memory is not the only constraint. One GPU has a fixed
throughput, so twenty tenants share one queue. Whether that is acceptable is a
throughput question, not a memory question, and it is what `bench/run_matrix.py`
is for. It was run, at concurrency 4 with two tenants, and the answer was that
multi-LoRA costs +9.3% on time-to-first-token and −10.0% on per-request token
rate against a base-only arm. `RESULTS.md` section (b).

## 5. Data privacy

Everything in this design lives inside the customer's own Azure subscription:

- **Training data** sits in the customer's Azure Blob Storage account.
- **Training** runs on a GPU in the customer's subscription.
- **The adapters** are written to the customer's Blob Storage.
- **The endpoint** is an Azure ML managed online endpoint in the customer's
  workspace, in the customer's region (`southcentralus` here).
- **The base model** is a public open-weights checkpoint pulled once at
  container start; it carries no customer data.

At inference time the request goes from the client to the customer's own scoring
URI and the response comes back. No prompt, no completion and no adapter weight
is sent to a third-party model API. There is no external inference vendor in the
request path at all. This is the main practical reason to run open weights on
your own hardware rather than call a hosted API: the data boundary is the
subscription boundary.

The one caveat worth stating plainly: pulling the base model at container start
does reach out to the Hugging Face model hub. That is an outbound fetch of public
weights, not an upload of anything. It can be removed entirely by baking the
weights into the container image or pre-staging them in Blob Storage, and the
serving README says how.

## 6. The request path

```
  client
    |
    |  POST https://<endpoint>.<region>.inference.ml.azure.com/v1/chat/completions
    |  Authorization: Bearer <key>
    |  { "model": "meridian",            <-- this field picks the tenant
    |    "messages": [ {"role":"system","content":"detailed thinking off"},
    |                  {"role":"user",   "content":"<the goal>"} ] }
    v
  +---------------------------------------------------------------+
  |  Azure ML managed online endpoint  (customer's subscription)   |
  |  - TLS termination, bearer-key auth, request routing           |
  +---------------------------------------------------------------+
    |
    v
  +---------------------------------------------------------------+
  |  container: vLLM OpenAI-compatible server, port 8000           |
  |                                                               |
  |   router: reads the "model" field                              |
  |     "base"     -> base weights only                            |
  |     "meridian" -> base weights + meridian adapter              |
  |     "vantage"  -> base weights + vantage adapter               |
  |                                                               |
  |   +-------------------------+   +---------------------------+  |
  |   |  FROZEN BASE  ~16 GB    |   |  LoRA slots (--max-loras) |  |
  |   |  loaded once, shared    | + |  meridian 160 MB          |  |
  |   |  by every request       |   |  vantage  160 MB          |  |
  |   +-------------------------+   +---------------------------+  |
  |                    |                                          |
  |                    v                                          |
  |          batched kernel: each row of the batch gets            |
  |          its own adapter applied to the shared base            |
  +---------------------------------------------------------------+
    |
    v
  tokens streamed back to the client
```

Adapter files reach the container from Blob Storage at container start (mounted
or downloaded), and are named to vLLM with `--lora-modules meridian=<path>
vantage=<path>`. The names on the left of the `=` are exactly the strings a
client puts in the `model` field.

## 7. What breaks

An honest list. None of these are hypothetical; all three are the normal failure
modes of this design.

**Adapter-cache thrashing.** `--max-loras` is the number of adapters that can be
in a single batch. If more distinct adapters are live at once than there are
slots, adapters get evicted and reloaded, and the cheap "pointer change" becomes
a repeated load. The symptom is latency that gets worse as tenant count rises,
not as request volume rises. The fix is to raise `--max-loras` and
`--max-cpu-loras` (which costs GPU and host memory respectively) or to shard
tenants across replicas. With two tenants and `--max-loras 4` this project has
headroom; a twenty-tenant deployment would need this tuned and measured.

**Cold-start latency on the first request per adapter.** This one did not
materialise here and the reason is worth knowing: statically registered adapters
are loaded at server start, so no request paid a load cost (section 3). It comes
back if you switch to dynamic adapter loading, or if enough distinct adapters are
live that eviction starts. The container's own cold start is real either way —
pulling and loading 16 GB of base weights took 2 min 26 s inside the container
and the deployment took 42.9 minutes end to end to accept traffic, which matters
for the endpoint's readiness probe and is discussed in `serve/azure/README.md`.

**One GPU is shared throughput.** Memory scales beautifully with tenant count.
Throughput does not. Every tenant's request queues behind every other tenant's on
the same device. A noisy tenant degrades everyone. There is no per-tenant
isolation of compute in this design — that would require separate endpoints,
which is exactly the cost the design is avoiding. If a tenant needs a latency
guarantee, they need their own replica, and the economics for that tenant revert
to the dedicated column.

A fourth, smaller one: quality. LoRA changes far less of the model than a full
fine-tune. If a tenant's task needs behaviour the base model cannot reach at all,
rank 16 over seven projections may not get there, and the answer is a higher rank
or a full fine-tune, not more adapters. `eval/separation.py` is the check for
this — it measures whether each adapter actually produces its own tenant's
contract and not the other's.

## 8. Alternative serving route: NVIDIA NIM

The documented alternative to running vLLM directly is **NVIDIA NIM for LLMs**,
which wraps the same idea in a supported container. NIM takes an environment
variable `NIM_PEFT_SOURCE` pointing at a directory of LoRA adapters, laid out as
one subdirectory per adapter containing `adapter_config.json` and
`adapter_model.safetensors` (or `adapter_model.bin`) — the standard Hugging Face
PEFT layout, which is exactly what `train/train_lora_hf.py` and
`train/convert_to_hf.py` produce. The subdirectory name becomes the adapter name
a client puts in the OpenAI `model` field, so the client contract is identical to
the vLLM route and none of the tools in this repo would change. NIM maintains a
host-memory PEFT cache and loads an adapter into it on first request, the same
cold/warm split described above; setting `NIM_PEFT_REFRESH_INTERVAL` makes NIM
poll the source directory so adapters can be added without a restart. NIM's own
documentation states that it passes `--enable-lora`, `--max-loras`,
`--max-cpu-loras` and `--max-lora-rank` through to vLLM, so the underlying
mechanism is the one described in section 3. The trade is support and packaging
against an NGC licence and less direct control over the flags. This project uses
vLLM directly because the flags are the thing being measured.

## Where the claims in this document get checked

| claim | checked by | outcome |
| --- | --- | --- |
| warm swap is effectively free | `bench/swap_time.py` (`warm_swap_estimate_s`) | confirmed: −48 ms / +4 ms |
| cold swap is a one-time per-adapter cost | `bench/swap_time.py` (`cold_swap_estimate_s`) | superseded: no runtime cold path, adapters preload at server start |
| adapters produce tenant-correct output | `eval/separation.py` (confusion matrix) | confirmed on 160 sealed goals: 100% own, 0% rival, base 0% both |
| one GPU carries both tenants at usable throughput | `bench/run_matrix.py` | yes, at a cost: +9.3% TTFT, −10.0% tokens/sec, 0 errors on 160 requests |
| the memory and cost arithmetic | `bench/economics.py` | re-run with the measured adapter size |
| adapter size on disk | training job artifact | measured: 167,832,240 bytes per tenant |
| per-adapter GPU footprint | `nvidia-smi` phase markers | **not measured** — invisible inside vLLM's pre-allocated pool |

Every number in this document that still carries a `~` is still an estimate.
`RESULTS.md` holds the measurements and names the log file behind each one.
