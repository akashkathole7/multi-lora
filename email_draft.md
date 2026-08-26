Send as a reply in the existing thread (where his brief and thumbs-up live),
not a new subject. Fill the two [bracketed] items before sending.

---

Monish,

Two tenant adapters on one A100, one Azure AI Foundry endpoint, measured.
Repo and one-pager below.

- Separation, on 160 goals hash-locked before training: each adapter passed
  its own tenant's contract 160/160 and the rival's 0/160; the base model
  passed neither on any goal. Verification is plain code, per our
  conversation — a deterministic Python checker, no model judged any output.
- Tenant switching: +53 ms p50 (alternating adapters vs repeating one),
  below the runs' own jitter. Adapters are GPU-resident from server start,
  so no request pays a load cost.
- Footprint: 0.168 GB per tenant against ~16 GB for a second full
  fine-tune. The whole project cost ~$35 of a $200 Azure credit.
- The honest limit: multi-LoRA costs +9.3% time-to-first-token and −10.0%
  tokens/sec versus base-only. RESULTS.md explains it.

Base is the dense 8B Nemotron Nano; the Nemotron 3 Nano port path is noted
in the repo. The endpoint is torn down to save credit — one script redeploys
it in ~30 minutes if you want to hit it live.

One-pager: [presentation share link]
Repo: https://github.com/akashkathole7/multi-lora — start with RESULTS.md.
Every number traces to a log file in the repo; the change log includes the
mistakes.

Happy to walk through it in 15 minutes, or answer anything async.

Aakash
[phone]
