# Results

Two LoRA adapters, one frozen 8B base, one A100, one endpoint. Every number below
comes out of a log file in this repository. The file that produced each number is
cited next to it. Where something was not measured, this document says
"not measured" rather than estimating it.

**Measured on:** Azure ML managed online endpoint `multilora-ep`, deployment `blue`,
SKU `Standard_NC24ads_A100_v4` (1x A100 80GB), region `southcentralus`, image
`vllm/vllm-openai:v0.27.1`, on 2026-08-25. Scoring URI was
`https://multilora-ep.southcentralus.inference.ml.azure.com/v1/chat/completions`.
The endpoint has been torn down and the deletion read back
(`serve/azure/logs/teardown_session1.log`).

---

## (a) Sealed separation matrix

160 goals held out of training, hash-locked before the run, three arms, 480
requests, concurrency 4, 0 HTTP errors, wall time 1151.65s.

Source: `eval/logs/separation_matrix_sealed.json`
(raw: `eval/logs/separation_raw_sealed.jsonl`, console:
`data/logs/sealed_run.log`).

| arm | passes Meridian rules | passes Vantage rules | n | errors |
| --- | ---: | ---: | ---: | ---: |
| base | 0.0% (0/160) | 0.0% (0/160) | 160 | 0 |
| meridian | 100.0% (160/160) | 0.0% (0/160) | 160 | 0 |
| vantage | 0.0% (0/160) | 100.0% (160/160) | 160 | 0 |

Goals file `eval/sealed_goals.jsonl`, sha256
`d54319eb6b8f78e314bec265ce052af8b3d1fe1192ddd5a15bffb4586d47fb15`, checked
against `eval/SEALED.sha256` before the first request was sent.

**Reading it.** The base row is the control and it is the important row. The base
model received the identical system message and the identical goal text, and it
satisfied neither tenant contract on any of 160 goals. So the behaviour is not
coming from *this* prompt. One control was not run and is listed in section (f):
a base arm carrying each tenant's house-style rules as a long system prompt. The
harness supports it unchanged; without it, this matrix proves the adapters beat
the deployed prompt, not that no prompt could close part of the gap. The two adapter rows are the diagonal: each adapter
satisfies its own tenant's contract on every goal and its rival's on none. The
off-diagonal zeros matter as much as the diagonal 160s — an adapter that had
merely learned "emit JSON" would score on both columns. Verification is
`data/verifier.py`, which is pure stdlib, has no model client and no network, so
the same response gets the same verdict on any machine. No model judged any
output.

For comparison, the dev-set matrix run earlier on 100 held-out goals
(`eval/logs/separation_matrix_20260825T103026225Z.json`, `reports/iter_02.md`):
base 0/0, meridian 100/100 own and 0/100 rival, vantage 98/100 own and 0/100
rival. The two vantage misses are itemised in section (f).

---

## (b) Four-arm serving benchmark

40 requests per arm, concurrency 4, 160 requests total, 0 errors, total wall time
291.60s. Prompt held constant across arms: "Cut unplanned downtime across our
three sites by 30% within a year." Driver is the stdlib fallback in
`bench/run_matrix.py`; genai-perf and vLLM's `benchmarks/benchmark_serving.py`
remain the preferred drivers and are still listed as unresolved in README's
"Unverified items". `max_tokens` is capped at 512 by the driver.

Source: `bench/logs/matrix_summary_endpoint_session1.json`
(raw: `bench/logs/matrix_raw_endpoint_session1.jsonl`, console:
`data/logs/bench_matrix_endpoint.log`). All 17 metric fields, all four arms:

| arm | concurrency | n_requests | errors | model_mix | ttft_p50_s | ttft_p95_s | itl_p50_s | itl_p95_s | tokens_per_sec_p50 | tokens_per_sec_p95 | tokens_per_sec_aggregate | requests_per_sec | e2e_p50_s | e2e_p95_s | wall_time_s | overhead_vs_base_pct |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base-only | 4 | 40 | 0 | base:40 | 1.123934 | 1.277523 | 0.011319 | 0.011642 | 69.451 | 74.274 | 269.929 | 0.714 | 5.558431 | 6.894963 | 56.055 | 0.0 |
| base-plus-one-lora | 4 | 40 | 0 | base:20 meridian:20 | 1.175948 | 1.737988 | 0.013248 | 0.013573 | 62.558 | 65.471 | 237.961 | 0.573 | 7.805531 | 8.207524 | 69.835 | 40.427 |
| two-lora-round-robin | 4 | 40 | 0 | vantage:20 meridian:20 | 1.306628 | 1.748656 | 0.013633 | 0.014029 | 61.769 | 63.373 | 245.696 | 0.48 | 8.288889 | 8.715142 | 83.355 | 49.123 |
| two-lora-interleaved | 4 | 40 | 0 | meridian:20 vantage:20 | 1.228817 | 1.426478 | 0.013628 | 0.014025 | 62.495 | 63.291 | 248.693 | 0.486 | 8.192628 | 8.498642 | 82.35 | 47.391 |

Definitions, from the same summary file: `itl` is the per-request mean
`(last_token_s - ttft_s)/(tokens-1)`, p50/p95 taken across requests;
`tokens_per_sec` per request is `output_tokens / e2e_s`;
`tokens_per_sec_aggregate` is `sum(output_tokens) / wall_time_s`;
`overhead_vs_base_pct` is `(arm e2e_p50 / base-only e2e_p50 - 1) * 100`.

All timings are **client-observed through the public Azure scoring URI**, so
TTFT includes network round-trip and endpoint-side queueing on top of prefill —
which is why an 8B model on an A100 shows ~1.1s TTFT here. Arms are compared
against each other over the same path, so the network component cancels in the
deltas. Mean output tokens per arm, from the raw log: base-only 378.3,
base-plus-one-lora 415.4, both two-lora arms 512.0 (the cap). The -32% req/s
against base is therefore a task-length effect — the adapters emit ~34% more
tokens per response — not a tenancy cost; normalized per token, the multi-LoRA
cost is the -10.0% above.

**The honest reading.** The project's stated aspiration for objective #2 was that
multi-LoRA serving cost under 5% overhead versus base
(`change_log.md` entry 14). It does not. Comparing `two-lora-interleaved`
against `base-only`:

| metric | base-only | two-lora-interleaved | delta |
| --- | ---: | ---: | ---: |
| TTFT p50 | 1.123934s | 1.228817s | +9.33% |
| TTFT p95 | 1.277523s | 1.426478s | +11.66% |
| tokens/sec per request p50 | 69.451 | 62.495 | -10.02% |
| tokens/sec aggregate | 269.929 | 248.693 | -7.87% |
| e2e p50 | 5.558431s | 8.192628s | +47.39% |

Time to first token is 9.3% worse. Per-request generation rate is 10.0% worse.
Both are roughly double the 5% target and they are reported as measured.

**The e2e number is not a third finding, and it is not cache thrashing.** The
+47% end-to-end figure decomposes into output length times per-token rate.
Median output length, computed from the raw log
`bench/logs/matrix_raw_endpoint_session1.jsonl`: base-only 381.5 tokens,
`two-lora-interleaved` 512 tokens. 512 is the driver's `max_tokens` cap, and both
adapter arms hit it on every request (min = max = 512 for all 40). The adapters
were trained to emit a complete structured JSON plan and they emit one; the base
model returns shorter prose. So:

    output-length ratio 512 / 381.5              = 1.342
    per-request rate ratio 69.451 / 62.495       = 1.111
    product                                       = 1.491
    observed e2e p50 ratio 8.192628 / 5.558431   = 1.474

Note that `tokens_per_sec` is defined as `output_tokens / e2e_s`, so this
decomposition is an accounting identity rather than an independent confirmation.
What it establishes is that the e2e gap is dominated by the adapters producing
more tokens, not by a collapse in serving rate. The honest cost of multi-LoRA on
this hardware is the +9.3% TTFT and the -10.0% token rate, not +47%.

**Cause.** LoRA's extra GEMMs per targeted projection. The server ran with
`--max-loras 4` and both adapters were resident throughout
(`serve/azure/logs/deployment_logs.txt` shows the launch line and no adapter
swap events), so this is not adapter-cache thrashing. `two-lora-round-robin` and
`two-lora-interleaved` differ only in request ordering and land within 0.1s of
each other on e2e p50, which is what you would expect if no eviction is
happening.

---

## (c) Adapter swap time

`bench/swap_time.py` run twice against a container no client had ever touched.
The deployment was restarted first (env-var nonce update, rolling reprovision)
specifically so the first-request sample would mean something — the earlier smoke
test had already warmed both adapters. Meridian ran first, so its first request
was the container's first-ever adapter request; vantage was still untouched when
its run began. 20 warm samples per distribution, 0 errors across 82 requests.

Sources: `bench/logs/swap_time_summary_20260825T122412363Z.json` (meridian) and
`bench/logs/swap_time_summary_20260825T122801312Z.json` (vantage), with raw
JSONL beside each; consoles `data/logs/swap_meridian_endpoint.log` and
`data/logs/swap_vantage_endpoint.log`.

| measurement | meridian run | vantage run |
| --- | ---: | ---: |
| first-request TTFT (n=1) | 0.976213s | 1.335302s |
| adapter warm TTFT p50 (n=20) | 1.232312s | 1.234827s |
| adapter warm TTFT p95 | 1.405424s | 1.418763s |
| base warm TTFT p50 (n=20) | 1.280591s | 1.230749s |
| base warm TTFT p95 | 1.572415s | 1.351416s |
| **warm swap = adapter p50 - base p50** | **-0.048279s** | **+0.004079s** |
| first-request minus adapter warm p50 | -0.256099s | +0.100475s |
| errors | 0 | 0 |
| wall time | 181.901s | 180.179s |

**Warm swap is zero within noise.** -48 ms and +4 ms, in opposite directions,
against a base p95-p50 spread of 292 ms on the same run. Sending a Meridian
request and a Vantage request back to back costs nothing measurable. That is the
expected result: the adapter is selected per row of the batch inside the kernel,
so switching tenants moves a pointer, not weights.

**Two definitions, reconciled.** `adapter p50 - base p50` compares a LoRA
request against a no-LoRA request, so it convolves the switch with the LoRA
compute tax of section (b) — that is why its sign flips run to run. The cleaner
isolation uses section (b)'s own arms, both LoRA-on: `two-lora-interleaved`
TTFT p50 (1.228817s, A/B alternating every request) minus `base-plus-one-lora`
TTFT p50 (1.175948s, the same adapter repeatedly) = **+52.9 ms**, against
p95-p50 spreads of 198-562 ms on those arms. Same verdict by either definition:
switching tenants costs nothing distinguishable from jitter, and the round-robin
arm ordering (`two-lora-round-robin`, a switch on every consecutive request)
lands within 0.1s of interleaved on e2e p50.

**There is no runtime cold path on this serving design.** Meridian's first-ever
request was *faster* than its own warm p50 (0.976s vs 1.232s). Vantage's was
100 ms slower than its warm p50, which is inside that run's warm p95-p50 spread
of 184 ms. Neither looks like a load. The cause is in the server launch line
(`serve/azure/logs/deployment_logs.txt`): adapters named in `--lora-modules` are
registered statically and loaded during server startup, so every tenant is
GPU-resident before the first client request arrives.

The Blob-to-GPU load cost is real, but it is paid once per container start, not
per request, and it is bounded by the startup timeline in section (d): the
container went from process start to `/health` answering in 2 min 26 s, adapters
included.

**Stated plainly: a per-request cold number exists only under dynamic adapter
loading (`VLLM_ALLOW_RUNTIME_LORA_UPDATING` plus the load/unload API), which is a
different serving mode from the one deployed. Not measured.**

---

## (d) Memory

`start_server.sh` samples `nvidia-smi --query-gpu=memory.used` at named phases and
every 30s, to stdout; the samples were recovered with
`az ml online-deployment get-logs`. Source:
`serve/azure/logs/deployment_logs.txt`.

| time (UTC) | phase | GPU memory used |
| --- | --- | ---: |
| 11:39:50Z | startup | 0 MiB |
| 11:40:20Z | after_base_download | 0 MiB |
| 11:40:50Z | periodic | 5 MiB |
| 11:41:20Z | periodic | 16,406 MiB |
| 11:41:50Z | periodic | 72,726 MiB |
| 11:42:16Z | server_ready_base_resident | 74,002 MiB |
| 11:42:16Z | before_first_request_meridian | 74,002 MiB |
| 11:42:16Z | after_first_request_meridian | 74,002 MiB |
| 11:42:16Z | before_first_request_vantage | 74,002 MiB |
| 11:42:16Z | after_first_request_vantage | 74,002 MiB |
| 11:42:16Z | steady_state | 74,002 MiB |
| 11:42:21Z - 11:56:51Z | periodic, 30 samples | 74,002 MiB (every one) |

The 16,406 MiB sample is the base weights landing. The jump to 72,726 MiB is
vLLM claiming its KV-cache pool.

**The caveat, stated before the number.** The before/after pairs around each
adapter's first request were designed to give a per-adapter GPU delta. They give
zero, and zero is not the adapter's cost. vLLM pre-allocates its memory pool at
`--gpu-memory-utilization 0.90` (the launch line in the same log) during startup,
so the pool is already claimed before any adapter is touched and `nvidia-smi`
cannot see anything that happens inside it. The per-adapter GPU footprint is
therefore **not measured by this instrument**, and no attempt is made here to
back it out.

**What is measured is the artifact.** The adapter weight file is
**167,832,240 bytes** (~160 MiB, ~0.168 GB), identical for both tenants by
construction, recorded in `change_log.md` entry 11 from the training job output.
That is fp32 storage of ~42M adapter parameters; stored bf16 it would be about
half (~0.084 GB), and the GPU-resident copy vLLM keeps is in the model's compute
dtype, so per-tenant GPU cost tracks the bf16 figure, not the fp32 file.

Against the counterfactual: a second full fine-tune of this base model is roughly
16 GB of weights (8B parameters at bf16). Per tenant, 0.168 GB against ~16 GB —
about 1%.

One discrepancy, now itemised. `start_server.sh` reported the mounted adapter
*directories* as 1,748,220,372 bytes (meridian) and 1,748,221,939 bytes
(vantage) in `serve/azure/logs/deployment_logs.txt` — roughly 10x the served
artifact. A blob-level listing of the registered asset
(`eval/logs/adapter_vantage_asset_listing.tsv`, 39 files summing to
1,748,205,555 bytes) shows why: the training-job output folder carries, beside
the root adapter (adapter_model.safetensors, 167,832,240 bytes — the only
tensors vLLM loads), TRL's per-epoch training checkpoints (`checkpoint-84/168/252`),
each holding its own copy of the adapter plus a 335,929,123-byte `optimizer.pt`
and tokenizer files. The 0.168 GB figure is the served per-tenant artifact and
is what section (e) uses; pruning `checkpoint-*/` from the asset before
registration would make the stored size match it. Shipped unpruned, the
per-tenant storage figure is 1.75 GB. Both numbers are in the logs.

---

## (e) Economics

Generated by `.venv/bin/python bench/economics.py --adapter-gb 0.168
--sku-price-usd-hr 3.673 --hours-month 730`. Output file:
`bench/logs/economics_measured_final.md`. The tool is arithmetic over named
constants, makes no network call, and says so on its first line. The adapter size
is the measured 0.168 GB from section (d); before this run the tool's default was
an unmeasured 0.08 GB estimate.

**Inputs** (from `bench/logs/economics_measured_final.md`):

| input | value | provenance |
| --- | --- | --- |
| base model weights | 16.00 GB | 8B params at bf16 |
| LoRA adapter | 0.1680 GB | measured artifact, section (d) |
| GPU SKU price | $3.673/hr | Azure pricing page, accessed 2026-08-24 |
| INR per USD | 87.00 | approximate |
| endpoint hours per month | 730.0 | `--hours-month` |
| endpoint cost per month | $2,681.29 | 3.673 x 730.0 |

**GPU memory for N tenants.** Column A is N separately fine-tuned models. Column
B is one base model plus N adapters. Weights only: KV cache, activations and CUDA
graphs sit on top of both columns and are the same for both.

| N tenants | A: N full fine-tunes (GB) | B: base + N adapters (GB) | GB saved | savings ratio A/B |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.00 | 16.17 | -0.17 | 0.99x |
| 2 | 32.00 | 16.34 | 15.66 | 1.96x |
| 3 | 48.00 | 16.50 | 31.50 | 2.91x |
| 4 | 64.00 | 16.67 | 47.33 | 3.84x |
| 5 | 80.00 | 16.84 | 63.16 | 4.75x |
| 6 | 96.00 | 17.01 | 78.99 | 5.64x |
| 7 | 112.00 | 17.18 | 94.82 | 6.52x |
| 8 | 128.00 | 17.34 | 110.66 | 7.38x |
| 9 | 144.00 | 17.51 | 126.49 | 8.22x |
| 10 | 160.00 | 17.68 | 142.32 | 9.05x |
| 11 | 176.00 | 17.85 | 158.15 | 9.86x |
| 12 | 192.00 | 18.02 | 173.98 | 10.66x |
| 13 | 208.00 | 18.18 | 189.82 | 11.44x |
| 14 | 224.00 | 18.35 | 205.65 | 12.21x |
| 15 | 240.00 | 18.52 | 221.48 | 12.96x |
| 16 | 256.00 | 18.69 | 237.31 | 13.70x |
| 17 | 272.00 | 18.86 | 253.14 | 14.43x |
| 18 | 288.00 | 19.02 | 268.98 | 15.14x |
| 19 | 304.00 | 19.19 | 284.81 | 15.84x |
| 20 | 320.00 | 19.36 | 300.64 | 16.53x |

At N=1 multi-LoRA is 0.17 GB worse than a dedicated fine-tune, and the table says
so. The crossover is immediate at N=2. The operationally interesting line is
N=5: 80 GB against 16.84 GB. An A100 has 80 GB, so the left column has run out of
one GPU by the fifth tenant while the right column has used a fifth of it.

**Cost per tenant per month.** Dedicated: each tenant on its own endpoint on its
own GPU. Shared: one endpoint serves all N, cost split evenly. The ratio between
the columns is exactly N by construction; the table exists to put an absolute
figure on it, not to discover a relationship.

| N tenants | dedicated USD/tenant/mo | shared USD/tenant/mo | dedicated INR/tenant/mo | shared INR/tenant/mo |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,681.29 | 2,681.29 | 233,272 | 233,272 |
| 2 | 2,681.29 | 1,340.64 | 233,272 | 116,636 |
| 3 | 2,681.29 | 893.76 | 233,272 | 77,757 |
| 4 | 2,681.29 | 670.32 | 233,272 | 58,318 |
| 5 | 2,681.29 | 536.26 | 233,272 | 46,654 |
| 6 | 2,681.29 | 446.88 | 233,272 | 38,879 |
| 7 | 2,681.29 | 383.04 | 233,272 | 33,325 |
| 8 | 2,681.29 | 335.16 | 233,272 | 29,159 |
| 9 | 2,681.29 | 297.92 | 233,272 | 25,919 |
| 10 | 2,681.29 | 268.13 | 233,272 | 23,327 |
| 11 | 2,681.29 | 243.75 | 233,272 | 21,207 |
| 12 | 2,681.29 | 223.44 | 233,272 | 19,439 |
| 13 | 2,681.29 | 206.25 | 233,272 | 17,944 |
| 14 | 2,681.29 | 191.52 | 233,272 | 16,662 |
| 15 | 2,681.29 | 178.75 | 233,272 | 15,551 |
| 16 | 2,681.29 | 167.58 | 233,272 | 14,580 |
| 17 | 2,681.29 | 157.72 | 233,272 | 13,722 |
| 18 | 2,681.29 | 148.96 | 233,272 | 12,960 |
| 19 | 2,681.29 | 141.12 | 233,272 | 12,277 |
| 20 | 2,681.29 | 134.06 | 233,272 | 11,664 |

**Assumptions, kept from the tool's own output.** Not included, because none of it
is measured: one-off fine-tuning cost per tenant (favours the adapter column, not
counted); whether one GPU has the throughput headroom for N tenants at the target
latency (favours the dedicated column — that is what section (b) starts to answer,
and it answers it only at concurrency 4 with two tenants); storage, egress, and
idle time outside the stated hours per month.

**On the price.** $3.673/hr is the Azure list price for
`Standard_NC24ads_A100_v4`, taken from the Azure Linux VM pricing page and
recorded in `bench/economics.py` with access date 2026-08-24; the table above was
re-run on 2026-08-25 with that constant unchanged. It is a **list price — verify
against the invoice.** Reserved instances, spot, enterprise agreements and
regional variation all move it, and every cost row moves proportionally with it.

**What this project actually spent.** Budget was a $200 Azure free-trial credit.
A fair objection the table should anticipate: N full fine-tunes could also
time-share one GPU, so the dedicated-GPU column is not the only alternative. But
time-sharing full models means a ~16 GB weight reload on every tenant switch and
no cross-tenant batching — requests for different tenants cannot share a forward
pass. The adapter design removes both costs: switching is the ~0 ms of section
(c), and heterogeneous tenants ride the same batch. That, not the storage line
alone, is the economic argument.

The one endpoint session is directly traceable: deployment `blue` created
2026-08-25 11:13:44Z (`serve/azure/logs/deploy_session1.log`), deleted 12:50:49Z
(`serve/azure/logs/teardown_session1.log`) — 1.618 hours, $5.94 at list price.
Documented waste, from `change_log.md`: a failed NeMo-route training job plus its
diagnostic, ~$1.70 (entry 9), and ~3.2 A100-hours serving zero requests because
of a YAML folded-scalar bug, ~$11.70 (entry 10 and its correction) — about $13.40
in total. Data generation with Azure OpenAI `gpt-5-mini` was projected at ~$3
(entry 6). Total project spend is estimated at roughly $35 including training
jobs and the dev-matrix eval job; that total is an estimate assembled from the
change log, **not** reconciled against an Azure invoice, and no invoice is in this
repository.

---

## (f) What breaks

Eleven limits. None are hypothetical.

**0a. The prompted-base control is missing.** The matrix's base arm ran with the
training-time system message, not with each tenant's house rules pasted in as a
long prompt. If a prompted base scored, say, 70%, the honest claim would become
"the adapter closes the last 30% and removes a ~2 KB per-request prompt" — still
a good story, but a different one. The harness runs this arm unchanged
(`eval/separation.py --served-names` plus a prompt variant); it was not run.

**0b. Runtime hot-add is supported, not demonstrated.** vLLM can load and unload
adapters on a live server (`VLLM_ALLOW_RUNTIME_LORA_UPDATING` + the
load/unload API); NIM exposes the same via its adapter store. This project served
statically registered adapters only, so "add a tenant without restarting" is a
documented capability here, not a measured one.

**1. The verifier checks vocabulary and schema, not semantics.** A pass means the
output has the right shape, uses the tenant's own terms and uses none of the
rival's. It does not mean the plan is good, feasible, or correct for the goal.
100% on the sealed matrix is 100% on tenant-style compliance. Anyone reading it
as "the model is right 100% of the time" is reading it wrong. What the design buys
is that the measurement is deterministic — no model judged any output, so the
number does not drift between runs, machines or graders.

**2. Two dev-set misses, and what they were.** On the 100-goal dev matrix the
vantage adapter passed 98/100. Both misses were goal-id 758 and goal-id 779,
recovered by re-running `data/verifier.py` over
`eval/logs/separation_raw_20260825T103026225Z.jsonl`. Both had `schema_ok: true`
and failed vocabulary only: 758 used the rival term "quarter/quarters", 779 used
"deviation/deviations". So the failure mode is a single leaked word in otherwise
correct output, not a structural collapse. Neither recurred on the sealed set,
which is a smaller sample of the same behaviour, not proof it is gone.

**3. One GPU is shared throughput.** Memory scales well with tenant count.
Throughput does not. Every tenant's request queues behind every other tenant's on
the same device, and there is no per-tenant compute isolation in this design. At
concurrency 4 with two tenants the endpoint delivered 0.486 req/s
(`bench/logs/matrix_summary_endpoint_session1.json`). A tenant that needs a
latency guarantee needs its own replica, and that tenant's economics revert to
the dedicated column in section (e).

**4. The LoRA compute tax is real: +9.3% TTFT, -10.0% tokens/sec.** Measured, in
section (b), against a 5% aspiration. It is the cost of the extra GEMMs and it
does not go away with tuning flags. Budget for it.

**5. Static registration bounds the resident tenant count.** The property that
makes swap free — adapters preloaded at server start — also means the number of
tenants is fixed at launch and bounded by `--max-loras` (4 here) and
`--max-cpu-loras` (8 here). Behaviour beyond those bounds — eviction, reload,
thrashing — was **not measured**. A 20-tenant deployment needs those flags tuned
and re-benchmarked, and adding a tenant needs a restart unless the deployment is
switched to dynamic loading, which is a different serving mode (section (c)).

**6. The sealed set is synthetic, from one generator.** All 1,600 training and
evaluation outputs were produced by a single model, Azure OpenAI `gpt-5-mini` at
`reasoning_effort: minimal` (`change_log.md` entry 6). Rejection rate on the full
set was 4.69%: 1,600 in, 1,525 kept, 75 rejected — 53 vocabulary, 19 parse, 3
schema (`data/generated/full/filter_summary.json`). The generator was never
trusted — every row passed the deterministic verifier before reaching training —
but one generator's idea of "Meridian voice" is still one generator's idea. The
goals are fictional and the tenants are fictional. Real customer data would
differ in ways this measurement cannot anticipate.

**7. The NeMo API route is dead in the 26.08 container.** The adapters were
trained with HF PEFT + TRL (`train/train_lora_hf.py`), inside
`nvcr.io/nvidia/nemo:26.08`, because that image's default interpreter ships
`megatron.bridge` but not `nemo` or `lightning` — NVIDIA moved the 26.x training
stack to Megatron-Bridge (`change_log.md` entry 9, evidence job
`funny_ball_xpnt0wxsxt`). The delivered adapters run in the NeMo-FW container and
serve identically, but they were not trained through the NeMo library. Anyone
reproducing this on a NeMo-API path needs to rewrite against Megatron-Bridge.

**8. The load driver is a fallback.** `bench/run_matrix.py` drives load with
stdlib `urllib.request`. genai-perf and vLLM's `benchmarks/benchmark_serving.py`
are the preferred generators and one of them produces a true per-token-pair ITL
distribution this driver cannot. The arms and the 17-field metric row are final;
the driver is not. Still listed under README's "Unverified items".

**9. One day, one region, one price list.** Every serving number here is from a
single deployment on 2026-08-25 in `southcentralus` on one
`Standard_NC24ads_A100_v4`. No repeat runs across days, no second region, no
second SKU, no variance estimate across deployments. Costs are list price. Treat
the numbers as one well-documented observation, not a distribution.

---

## (g) Provenance

Nothing in this file was typed from memory.

Each number above names the file it came from: the sealed matrix from
`eval/logs/separation_matrix_sealed.json`; the benchmark rows from
`bench/logs/matrix_summary_endpoint_session1.json`; output-token medians
recomputed from `bench/logs/matrix_raw_endpoint_session1.jsonl`; swap numbers
from the two `bench/logs/swap_time_summary_20260825T12*.json` files; GPU memory
from `serve/azure/logs/deployment_logs.txt`; deployment and teardown timings from
`serve/azure/logs/deploy_session1.log` and
`serve/azure/logs/teardown_session1.log`; data-filter numbers from
`data/generated/full/filter_summary.json`; the economics tables from
`bench/logs/economics_measured_final.md`, which `bench/economics.py` wrote from
the command in section (e); the two dev-set misses recovered by re-running
`data/verifier.py` over `eval/logs/separation_raw_20260825T103026225Z.jsonl`.

Every measurement tool writes raw per-request JSONL first and computes its
summary by reading that file back off disk, so each summary carries the path of
the log it came from. `eval/separation.py` and `bench/*.py` import no model
client; `scripts/check_no_model_imports.sh` fails the build if one ever appears
in `data/verifier.py` or under `eval/`. The measurement harness was built and
self-tested against a mock server before any adapter existed
(`change_log.md` entry 3), so the correct answers were fixed by construction
before there was a result to attach to them.

`change_log.md` holds 17 entries and is the project's memory. It records the
mistakes at the same resolution as the results: the training route that had to be
abandoned (entry 9), the YAML folded-scalar bug that billed an A100 for 3.2 hours
to serve zero requests (entry 10 and its correction), a cancel command that
silently failed and was reported as done, evidence files silently dropped from a
commit by a gitignore rule (entry 16 correction), and a commit message with the
wrong stage label (entry 3). About $13.40 of the spend is documented waste. It is
in the log because a result you cannot audit is not a result.

Where a number does not exist, this document says so: per-adapter GPU footprint
(not measurable through a pre-allocated pool), per-request cold-load latency (not
measured — wrong serving mode), thrashing beyond `--max-loras` (not measured),
and the total project invoice (estimated, not reconciled).
