# multi-lora

Two LoRA adapters, two fictional tenants, one task. Both tenants turn a leadership-intent
statement into a structured JSON execution plan, but they have opposite house styles.
Meridian Industrial is a regulated manufacturer: formal, gate reviews, compliance notes.
Vantage Cloud is a fast SaaS company: terse, OKRs, sprints, owners. A deterministic
pure-Python verifier (`data/verifier.py`) enforces both contracts. It is the data filter
during generation, the gate on what reaches training, and the measurement at the end.
It has no model client and no network dependency, so the same code produces the same
verdict on any machine.

Stage 0 is the verifier, six fixture cases with fixed expected verdicts, the staged data
pipeline, and a 10-goal dry run that exercises the pipeline plumbing on hand-authored
fixtures. Stage 1 adds the measurement tooling — the separation harness, the swap-time
and serving benchmarks, the economics table and the report generator — plus a mock
server that makes all of it self-testable offline. No model has been called in this repo
and no adapter has been trained.

## Layout

```
data/verifier.py       deterministic schema + vocabulary checker (stdlib only)
data/test_outputs.py   the fixture verdict table, pytest or plain python
data/generate.py       goals -> outputs -> filter -> package
data/fixtures/         six verdict fixtures + the dry-run fixtures
data/logs/             console output for every check in this stage
tools/                 mock vLLM OpenAI server, for running the tools offline
eval/separation.py     the proof artifact: per-arm tenant confusion matrix
bench/                 swap_time.py, run_matrix.py, economics.py
scripts/               guardrail + make_report.py
tests/                 self-test for the measurement tooling
train/ serve/          later stages, empty
```

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
adapter that stands in for a cold adapter load. Later the same tools point at the
real Azure scoring URI by changing `--endpoint`; nothing else changes.

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
against the mock server. **No model was called and no adapter exists yet.** Those
numbers describe the measuring instrument, not the system under test, and
`iter_00.md` says so on its first line. The adapter size of 0.08 GB in
`bench/economics.py` is likewise an estimate, not a measurement, until Stage 2;
re-run with `--adapter-gb <measured>` then.

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
run unless `--allow-rerun` is given. No sealed set exists yet — it is created at
Stage 3, against goals held out of training.

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

- Every number in this repo traces to a log file under `data/logs/`. Anything synthetic
  or fixture-based is labelled as such where it appears, including a `_note` marker on
  the first line of both dry-run files.
- `ANTHROPIC_API_KEY` is not present in the build environment. Stage 1 real data
  generation is blocked on it. The Stage 0 dry run validates pipeline plumbing on
  labelled fixtures only, and no fixture in this repo is training data.
- The local machine has no usable GPU (GTX 1650, 4GB). All training and serving stages
  run on Azure. The az CLI is not installed locally yet.
- Base model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. The system message
  `detailed thinking off` goes in every training row and every eval call.
- Serving will be vLLM with `--enable-lora` behind an Azure ML managed endpoint,
  OpenAI-compatible, bearer key in an `Authorization` header. Adapter selection is
  per-request via the `model` field: `base`, `meridian`, `vantage`. Every tool takes
  `--served-names` so those strings can change without a code edit.
- `bench/swap_time.py` is the only hand-rolled timing tool, and deliberately so: a
  cold adapter load happens once per adapter per server lifetime, and isolating it
  needs control over request order that a load generator does not give you.
  `bench/run_matrix.py` fixes the arms and the metric row but expects to hand the
  driving to genai-perf or vLLM's `benchmark_serving.py` at Stage 3.
- `blockers[]` may be empty. The spec is silent on a minimum, so the verifier requires
  the key to be present and to be a list of non-empty strings, and accepts an empty list.
- Unknown top-level keys are rejected for both tenants. Strictness was chosen over
  tolerance because the verifier has to give the same answer every time it runs.
