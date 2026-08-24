# Multi-tenant serving economics (run selftest)

Arithmetic only. No measurement, no network call. Produced by `bench/economics.py`; every number below is a function of the inputs in the next table and nothing else.

## Inputs

| input | value | provenance |
| --- | --- | --- |
| base model weights | 16.00 GB | 8B params at bf16; replace with the measured checkpoint size at Stage 2 |
| LoRA adapter | 0.0800 GB | **ESTIMATE** - rank-16 LoRA on 8B across 7 target modules. NOT MEASURED. Re-run with `--adapter-gb <measured>` at Stage 2. |
| GPU SKU price | $3.673/hr | Azure pricing page, accessed 2026-08-24; verify at Stage 4 |
| INR per USD | 87.00 | approximate; verify at Stage 4 |
| endpoint hours per month | 730 | `--hours-month` |
| endpoint cost per month | $2,681.29 | 3.673 x 730 |

## GPU memory for N tenants

Column A is N separately fine-tuned models. Column B is one base model plus N adapters. Weights only: KV cache, activations and CUDA graphs are on top of both columns and are the same for both.

| N tenants | A: N full fine-tunes (GB) | B: base + N adapters (GB) | GB saved | savings ratio A/B |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.00 | 16.08 | -0.08 | 1.00x |
| 2 | 32.00 | 16.16 | 15.84 | 1.98x |
| 3 | 48.00 | 16.24 | 31.76 | 2.96x |
| 4 | 64.00 | 16.32 | 47.68 | 3.92x |
| 5 | 80.00 | 16.40 | 63.60 | 4.88x |
| 6 | 96.00 | 16.48 | 79.52 | 5.83x |
| 7 | 112.00 | 16.56 | 95.44 | 6.76x |
| 8 | 128.00 | 16.64 | 111.36 | 7.69x |
| 9 | 144.00 | 16.72 | 127.28 | 8.61x |
| 10 | 160.00 | 16.80 | 143.20 | 9.52x |
| 11 | 176.00 | 16.88 | 159.12 | 10.43x |
| 12 | 192.00 | 16.96 | 175.04 | 11.32x |
| 13 | 208.00 | 17.04 | 190.96 | 12.21x |
| 14 | 224.00 | 17.12 | 206.88 | 13.08x |
| 15 | 240.00 | 17.20 | 222.80 | 13.95x |
| 16 | 256.00 | 17.28 | 238.72 | 14.81x |
| 17 | 272.00 | 17.36 | 254.64 | 15.67x |
| 18 | 288.00 | 17.44 | 270.56 | 16.51x |
| 19 | 304.00 | 17.52 | 286.48 | 17.35x |
| 20 | 320.00 | 17.60 | 302.40 | 18.18x |

## Cost per tenant per month

Dedicated: each tenant runs its own endpoint on its own GPU. Shared: one endpoint serves all N, cost split evenly. The ratio between the two columns is exactly N by construction - the table is here to put an absolute figure on it, not to discover a relationship.

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

## Not included

- One-off fine-tuning cost per tenant. Favours column B; not counted.
- Whether one GPU has the throughput headroom for N tenants at the target latency. Favours the dedicated column; `bench/run_matrix.py` measures it.
- Storage, egress, and idle time outside the stated hours per month.

