DRAFT — Aakash reviews before sending. Not sent.

Subject: Multi-LoRA on one A100 — measured results

Monish,

I trained two tenant-specific LoRA adapters on Llama-3.1-Nemotron-Nano-8B —
HF PEFT route inside NVIDIA's NeMo-FW container, because the 26.08 image no
longer ships the NeMo training API — and served both from one A100 behind an
Azure ML (Foundry) managed online endpoint.

Two numbers.

Separation, on 160 goals held out of training and hash-locked before the run:
each adapter satisfied its own tenant's contract 160/160 and the rival's 0/160.
The base model, same prompt, passed neither on any goal. Verification is plain
code, per our conversation — a deterministic Python checker decides; no model
judged any output.

Tenant switching: +4 ms and −48 ms against baseline, zero within noise. Both
tenants are GPU-resident from server start, so no request pays a load cost.

Economics: 0.168 GB per tenant against ~16 GB for a second full fine-tune. The
project cost about $35 of a $200 Azure credit; the endpoint is torn down.

One limit, measured not assumed: multi-LoRA costs +9.3% time-to-first-token and
−10.0% tokens/sec versus base-only. RESULTS.md explains it.

One-page summary with the tables and the architecture diagram: <PRESENTATION_URL>
Repo: <REPO_URL>. Every number traces to a log file in the repo; the change log
includes the mistakes.

Aakash
