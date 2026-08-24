# multi-lora

Two LoRA adapters, two fictional tenants, one task. Both tenants turn a leadership-intent
statement into a structured JSON execution plan, but they have opposite house styles.
Meridian Industrial is a regulated manufacturer: formal, gate reviews, compliance notes.
Vantage Cloud is a fast SaaS company: terse, OKRs, sprints, owners. A deterministic
pure-Python verifier (`data/verifier.py`) enforces both contracts. It is the data filter
during generation, the gate on what reaches training, and the measurement at the end.
It has no model client and no network dependency, so the same code produces the same
verdict on any machine.

Stage 0 is what is here now: the verifier, six fixture cases with fixed expected verdicts,
the staged data pipeline, and a 10-goal dry run that exercises the pipeline plumbing on
hand-authored fixtures. No model has been called in this repo.

## Layout

```
data/verifier.py       deterministic schema + vocabulary checker (stdlib only)
data/test_outputs.py   the fixture verdict table, pytest or plain python
data/generate.py       goals -> outputs -> filter -> package
data/fixtures/         six verdict fixtures + the dry-run fixtures
data/logs/             console output for every check in this stage
scripts/               guardrail: verifier and eval/ must not import a model client
train/ serve/ bench/ eval/ reports/   later stages, empty
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
- `blockers[]` may be empty. The spec is silent on a minimum, so the verifier requires
  the key to be present and to be a list of non-empty strings, and accepts an empty list.
- Unknown top-level keys are rejected for both tenants. Strictness was chosen over
  tolerance because the verifier has to give the same answer every time it runs.
