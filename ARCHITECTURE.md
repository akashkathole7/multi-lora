# Architecture

This document explains how two customers ("tenants") get their own custom model
behaviour out of one GPU. It is written for a reader who does not work with
machine learning day to day. There is no marketing here; where a number is an
estimate rather than a measurement, it says so.

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

**Size estimate.** Assuming the Llama-3.1-8B geometry (32 layers, hidden size
4096, feed-forward size 14336, 8 key/value heads), rank 16 over those seven
modules comes to about 42 million adapter parameters. At bf16 that is roughly
**84 MB on disk** — call it 40–90 MB depending on how the checkpoint is stored
(fp32 storage roughly doubles it; storing fewer modules roughly halves it).

> This is an **ESTIMATE**, not a measurement. It is arithmetic over the
> published layer dimensions, and the exact geometry of the Nemotron Nano
> variant has not been confirmed against its config file. `bench/economics.py`
> uses 0.08 GB as its default adapter size for the same reason and labels every
> table it prints as ESTIMATE. The number is replaced by `du -b` on the real
> adapter directory once training has run.

The comparison that matters: **16,000 MB of base model, 84 MB of tenant.** The
tenant-specific part is about half a percent of the whole.

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

**Cold swap — the first time only.** The first request for an adapter the server
has not served yet has to fetch that adapter's weights and put them on the GPU:
Azure Blob Storage → the container's local disk → host memory → GPU memory. That
is a one-time cost per adapter per server lifetime, of the order of a fraction of
a second for an 84 MB file, and after it the adapter is warm. `--max-cpu-loras`
sets how many adapters are kept in host memory, which shortens a re-load if an
adapter is evicted from the GPU slots and needed again.

`bench/swap_time.py` measures both by controlling request order: warm baseline
first, then exactly one first-touch request to the adapter (the cold sample),
then repeated requests to the same adapter (the warm samples). Cold minus warm is
the adapter load cost.

## 4. The economics of many tenants

| tenants | separate fine-tuned models | one base + N adapters |
| ---: | ---: | ---: |
| 1 | ~16 GB | ~16.08 GB |
| 2 | ~32 GB | ~16.16 GB |
| 5 | ~80 GB | ~16.40 GB |
| 20 | ~320 GB | ~17.60 GB |

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
is for.

## 5. Data privacy

Everything in this design lives inside the customer's own Azure subscription:

- **Training data** sits in the customer's Azure Blob Storage account.
- **Training** runs on a GPU in the customer's subscription.
- **The adapters** are written to the customer's Blob Storage.
- **The endpoint** is an Azure ML managed online endpoint in the customer's
  workspace, in the customer's region (East US here).
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
    |  POST https://<endpoint>.<region>.inference.ml.azure.com/score
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
  |   |  loaded once, shared    | + |  meridian ~84 MB          |  |
  |   |  by every request       |   |  vantage  ~84 MB          |  |
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

**Cold-start latency on the first request per adapter.** The first request to
each adapter after a server restart pays the load cost. A tenant whose traffic is
one request an hour may pay it repeatedly if the adapter keeps getting evicted.
Mitigations: warm every adapter with a synthetic request at startup, or keep
`--max-cpu-loras` high enough that eviction only goes as far as host memory.
Separately, the whole container has a cold start of its own — pulling and loading
16 GB of base weights takes minutes, which matters for the endpoint's readiness
probe and is discussed in `serve/azure/README.md`.

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

| claim | checked by |
| --- | --- |
| warm swap is effectively free | `bench/swap_time.py` (`warm_swap_estimate_s`) |
| cold swap is a one-time per-adapter cost | `bench/swap_time.py` (`cold_swap_estimate_s`) |
| adapters produce tenant-correct output | `eval/separation.py` (confusion matrix) |
| one GPU carries both tenants at usable throughput | `bench/run_matrix.py` |
| the memory and cost arithmetic | `bench/economics.py` |
| adapter size on disk | not yet measured — Stage 2 |

Until those have run against a real endpoint, every number in this document that
carries a `~` is an estimate.
