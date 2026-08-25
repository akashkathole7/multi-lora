# Training

Two routes to the same artifact: a rank-16 LoRA adapter per tenant, in Hugging
Face PEFT layout, ready for `--lora-modules`.

| | route A (primary) | route B (fallback) |
| --- | --- | --- |
| script | `train_lora.py` | `train_lora_hf.py` |
| stack | NeMo Framework 2.x | HF PEFT + TRL `SFTTrainer` |
| environment | `nvcr.io/nvidia/nemo` container | `pip install torch transformers peft trl datasets` |
| output | `.nemo` checkpoint, then `convert_to_hf.py` | HF PEFT adapter directly |
| where | 1x A100 on Azure | 1x A100 on Azure |

Both read the **same** `config_<tenant>.yaml`. Rank, alpha, dropout, target
modules, seed, epochs, learning rate, schedule, batch sizes and sequence length
all live in that one file. Two routes with separately typed hyperparameters
would be two different experiments, and a fallback that produces an
incomparable artifact is not a fallback.

## The config files are not NeMo config files

NeMo 1.0 took Hydra YAML. NeMo 2.0 does not — "NeMo 2.0 shifts to a
Python-based configuration"
([docs](https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemo-2.0/index.html)).
So `config_meridian.yaml` and `config_vantage.yaml` are this project's own
config surface. `train_lora.py` reads them and calls the NeMo Python API.

The one place this shows up as a real difference: **NeMo calls the LoRA rank
`dim`; PEFT calls it `r`.** Same number, two spellings, and the single most
likely place for the two routes to drift apart. Both scripts print the value
they resolved before doing anything.

## Target modules differ between the routes, on purpose

The config lists seven Hugging Face module names. That is the portable form —
it is what PEFT takes verbatim and what ends up in the exported
`adapter_config.json` that vLLM and NIM read.

NeMo does not accept those names. Megatron fuses projections, so `train_lora.py`
translates:

```
q_proj, k_proj, v_proj  ->  linear_qkv     (one fused adapter, not three)
o_proj                  ->  linear_proj
gate_proj, up_proj      ->  linear_fc1     (one fused adapter, not two)
down_proj               ->  linear_fc2
```

The NeMo docs are explicit about the consequence: "the Hugging Face
implementation is equivalent to NeMo's CanonicalLoRA, not LoRA. However both can
be converted to the Hugging Face implementation."
([docs](https://docs.nvidia.com/nemo-framework/user-guide/25.09/sft_peft/peft_nemo2.html))

So route A with the default `nemo_use_canonical: false` trains slightly fewer,
slightly larger adapters than route B. Set `nemo_use_canonical: true` to match
route B exactly. `convert_to_hf.py --verify` prints the `target_modules` that
actually came out, so this is visible rather than assumed.

## When the fallback triggers

Route B is used when NeMo fights the environment. Concretely, any of:

- **The base model's NeMo config class does not resolve.** The Llama Nemotron
  docs publish an `import_ckpt` example for the Ultra 253B variant, not the
  Nano 8B. `train_lora.py` refuses to guess a substitute and prints the
  candidates it can see in the container.
- **The container tag does not match the docs.** NeMo has been moving toward
  Megatron-Bridge, whose API differs from the `nemo.collections.llm` calls this
  script makes.
- **Megatron's parallel setup will not initialise on a single GPU** without
  more configuration than the budget justifies.
- **The chat-template question cannot be settled.** Route A must flatten chat
  rows into NeMo's `{"input", "output"}` shape and then trust NeMo to template
  them. Route B feeds the chat rows in unchanged and TRL applies the
  tokenizer's own chat template — the same one vLLM applies at serving time.
  That prompt-shape risk does not exist on route B at all, which is the one
  place the fallback is genuinely better than the primary.
- **Time.** The budget is about $115 total. An A100 costs roughly $3.67/hour.
  Debugging a container is billed at the same rate as training.

**A route switch goes through `change_log.md`.** Not a commit message, not a
comment — a numbered entry saying which route was used, what made the other one
unusable, and what that changes about the resulting adapter. Two adapters
trained by different routes are not interchangeable evidence, and six weeks
later nobody remembers which one produced which number.

## Running it

Nothing here runs on the build machine. Both scripts have a `--dry-run` that
does, and it is worth using first: it resolves the config, converts the data,
prints the exact API calls, and imports nothing heavy.

```bash
# anywhere, including a laptop
python train/train_lora.py    --tenant meridian --dry-run
python train/train_lora_hf.py --tenant meridian --dry-run
```

### Route A, NeMo

```bash
docker run --gpus all --ipc=host \
    -v "$PWD":/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.09.02 \
    python train/train_lora.py --tenant meridian

python train/convert_to_hf.py --tenant meridian     # .nemo -> HF PEFT layout
```

### Route B, HF PEFT + TRL

```bash
pip install torch transformers peft trl datasets accelerate
python train/train_lora_hf.py --tenant meridian     # writes HF PEFT directly
```

### Either way, verify before serving

```bash
python train/convert_to_hf.py --verify train/out/meridian_hf
```

That checks `adapter_config.json` and `adapter_model.safetensors` exist, prints
the rank, alpha and target modules that actually came out, cross-checks them
against the training config, and reports the on-disk size. The size matters
beyond bookkeeping: it is the measurement that retires the 0.08 GB **ESTIMATE**
in `bench/economics.py` and in `ARCHITECTURE.md`.

The rank check matters too. vLLM's `--max-lora-rank` must be greater than or
equal to the adapter's `r`, or the adapter is rejected at load time — after the
endpoint is already billing.

## Layout

```
train/config_meridian.yaml   hyperparameters, tenant meridian
train/config_vantage.yaml    hyperparameters, tenant vantage
train/train_lora.py          route A: NeMo 2.x PEFT
train/train_lora_hf.py       route B: HF PEFT + TRL SFTTrainer
train/convert_to_hf.py       .nemo -> HF PEFT, and --verify for either route
train/logs/                  per-run logs, written by both routes
train/out/<tenant>_nemo/     route A checkpoint
train/out/<tenant>_hf/       HF PEFT adapter (both routes end up here)
train/out/adapters/          staging dir deploy.sh uploads and registers
```

## Data

Both routes read `data/generated/train_<tenant>.jsonl`, written by
`python data/generate.py package`. One JSON object per line:

```json
{"messages": [{"role": "system",    "content": "detailed thinking off"},
              {"role": "user",      "content": "<the leadership goal>"},
              {"role": "assistant", "content": "<the tenant's JSON plan>"}]}
```

The system message is exactly `detailed thinking off` on every row, and it is
also on every eval call in `eval/separation.py`. Route B checks this and warns
on any row that does not match; route A prepends the system message to the
`input` field rather than dropping it. If training and serving disagree about
the prompt shape, the adapter is fitted to a prompt that never occurs and the
separation number is quietly low with no visible error.

**As of this commit there is no training data.** Stage 1 is blocked on
`ANTHROPIC_API_KEY`. Both scripts exit 2 with that message rather than
half-running.
