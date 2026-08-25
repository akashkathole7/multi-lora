#!/usr/bin/env python3
"""PEFT LoRA fine-tuning for one tenant, via the NVIDIA NeMo Framework 2.x API.

EXPECTED ENVIRONMENT
--------------------
This script does not run on the build machine and is not meant to. It runs
inside the NVIDIA NeMo Framework container, on a single A100:

    docker run --gpus all --ipc=host \\
        -v "$PWD":/workspace -w /workspace \\
        nvcr.io/nvidia/nemo:25.09.02 \\
        python train/train_lora.py --tenant meridian

    # CHECK: container tag. 25.09.02 is the tag whose user guide documents the
    # CHECK: `nemo.collections.llm` 2.x API this file calls
    # CHECK: (https://docs.nvidia.com/nemo-framework/user-guide/25.09/). NGC tag
    # CHECK: listings were not reachable to confirm it is the newest available,
    # CHECK: and newer NeMo releases have been moving toward Megatron-Bridge,
    # CHECK: whose API differs. Pin whatever tag you verify on
    # CHECK: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags and
    # CHECK: record it in change_log.md.

Hardware assumed: 1x A100 80GB, which is what serve/azure runs on. LoRA on an
8B model at seq length 2048 and micro batch 1 fits comfortably; the memory
headroom is why micro_batch_size is 1 and gradient accumulation carries the
global batch of 8.

WHY THERE IS A YAML FILE IF NEMO 2.x IS PYTHON-CONFIGURED
---------------------------------------------------------
NeMo 1.0 took Hydra YAML. NeMo 2.0 does not: "In NeMo 1.0, the main interface
for configuring experiments is through YAML files... NeMo 2.0 shifts to a
Python-based configuration."
  https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemo-2.0/index.html

train/config_<tenant>.yaml is therefore THIS PROJECT'S config surface, not
NeMo's. This script reads it and calls the NeMo Python API. The reason to keep
a YAML file at all is that train_lora_hf.py — the fallback route — reads the
same file, which is the only way to guarantee the two routes train on identical
hyperparameters. A hyperparameter that lives in one script and is retyped into
the other is a hyperparameter that will eventually differ.

WHAT IS VERIFIED AND WHAT IS NOT
--------------------------------
Verified against the NeMo 2.x docs (2026-08-24), URL at each call site below:
  llm.peft.LoRA(dim=, alpha=, dropout=, target_modules=)   <- rank is `dim`
  llm.finetune(model=, data=, trainer=, peft=, optim=, log=)
  llm.import_ckpt(model=, source='hf://<id>')
  llm.FineTuningDataModule(dataset_root=, seq_length=, ...)
  nl.MegatronMixedPrecision(precision="bf16-mixed")
  nl.MegatronOptimizerModule(config=OptimizerConfig(...), lr_scheduler=...)
  nl.lr_scheduler.CosineAnnealingScheduler(...)

Everything the docs did not settle is marked `# CHECK:` at the exact line and
listed in README.md. The largest one is the base model's NeMo config class
name. This script REFUSES TO RUN on a name it cannot resolve rather than
substituting a plausible one — training the wrong architecture for three epochs
on a billable A100 is the expensive failure mode here.

Run `--dry-run` first. It resolves the config, converts the data, prints the
exact NeMo calls it would make, and never imports NeMo. That works on any
machine, including this one.

Examples:
  python train/train_lora.py --tenant meridian --dry-run
  python train/train_lora.py --tenant vantage --config train/config_vantage.yaml
  python train/train_lora.py --tenant meridian --prepare-data-only
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TENANTS = ("meridian", "vantage")

# HF projection name -> NeMo/Megatron fused module name.
#
# Megatron fuses projections that Hugging Face keeps separate, so the seven HF
# names in the config collapse to four NeMo names. Documented NeMo values are
# linear_qkv / linear_proj / linear_fc1 / linear_fc2:
#   https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.peft.module_matcher.html
#
# The NeMo PEFT guide is explicit about what this costs: "the Hugging Face
# implementation is equivalent to NeMo's CanonicalLoRA, not LoRA. However both
# can be converted to the Hugging Face implementation."
#   https://docs.nvidia.com/nemo-framework/user-guide/25.09/sft_peft/peft_nemo2.html
#
# Practically: with LoRA, q/k/v share ONE adapter over the fused qkv matrix;
# with CanonicalLoRA they get three, matching HF. Set nemo_use_canonical: true
# in the config to match HF exactly. Default is false because fused LoRA is
# NeMo's own recommended default and is faster.
HF_TO_NEMO_MODULE = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    try:
        import yaml  # noqa: PLC0415 - keep the import failure message specific
    except ImportError:
        raise SystemExit(
            "PyYAML is not installed. pip install pyyaml\n"
            "(It is present in the NeMo container and in this repo's .venv.)"
        )
    if not path.exists():
        raise SystemExit(f"no config at {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: top level is not a mapping")
    return config


def resolve_path(value: str) -> Path:
    """Config paths are repo-relative unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def translate_target_modules(hf_modules: list) -> list:
    """HF projection names -> the deduplicated NeMo fused names, order kept."""
    out = []
    for name in hf_modules:
        nemo_name = HF_TO_NEMO_MODULE.get(name)
        if nemo_name is None:
            raise SystemExit(
                f"target module {name!r} has no known NeMo equivalent. "
                f"Known: {', '.join(sorted(HF_TO_NEMO_MODULE))}"
            )
        if nemo_name not in out:
            out.append(nemo_name)
    return out


# --------------------------------------------------------------------------
# data conversion: chat JSONL -> what FineTuningDataModule expects
# --------------------------------------------------------------------------


def read_chat_jsonl(path: Path) -> list:
    """Read {"messages": [...]} rows, skipping the provenance marker line."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON: {exc}")
            if "messages" not in row:
                continue  # provenance marker row, as written by data/generate.py
            rows.append(row)
    return rows


def split_messages(row: dict) -> tuple:
    """Pull (system, user, assistant) out of one chat row."""
    by_role = {}
    for message in row.get("messages", []):
        by_role[message.get("role")] = message.get("content", "")
    return by_role.get("system", ""), by_role.get("user", ""), by_role.get("assistant", "")


def convert_to_nemo_jsonl(chat_rows: list, validation_fraction: float, seed: int) -> tuple:
    """Chat rows -> NeMo {"input", "output"} rows, split train/validation.

    NeMo's FineTuningDataModule expects a dataset_root holding training.jsonl,
    validation.jsonl and test.jsonl, whose rows are {"input": ..., "output": ...}
    — NOT OpenAI chat rows:
      https://docs.nvidia.com/nemo-framework/user-guide/25.09/data/finetune_data.html

    CHECK: how NeMo wraps `input` in the model's chat template before
    CHECK: tokenising is not documented for this data module, and it matters
    CHECK: more than it looks. Every eval call in this repo sends a real chat
    CHECK: request with system "detailed thinking off" and the goal as the user
    CHECK: turn. If training sees a differently-templated prompt, the adapter is
    CHECK: fitted to a prompt shape that never occurs at inference and the
    CHECK: separation number will be quietly low. Inspect the tokenised first
    CHECK: batch inside the container before committing to a 3-epoch run, and
    CHECK: record what the template turned out to be in change_log.md.
    CHECK: NeMo also ships a ChatDataModule ("sets a few default arguments on
    CHECK: top of FineTuningDataModule") whose expected JSONL schema is not
    CHECK: published. If it accepts {"messages": [...]} directly, this whole
    CHECK: conversion should be deleted in favour of it.
    """
    import random  # noqa: PLC0415 - only needed here

    converted = []
    for row in chat_rows:
        system, user, assistant = split_messages(row)
        if not user or not assistant:
            continue
        # System message is prepended rather than dropped: "detailed thinking
        # off" is part of every serving-time prompt in this project, so the
        # adapter must be trained with it present.
        prompt = f"{system}\n\n{user}" if system else user
        converted.append({"input": prompt, "output": assistant})

    rng = random.Random(seed)
    rng.shuffle(converted)
    n_val = max(1, int(len(converted) * validation_fraction)) if converted else 0
    return converted[n_val:], converted[:n_val]


def write_nemo_dataset(dataset_root: Path, train_rows: list, val_rows: list) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list) -> None:
        with (dataset_root / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump("training.jsonl", train_rows)
    dump("validation.jsonl", val_rows)
    # FineTuningDataModule looks for all three names. Test is unused here; the
    # real held-out measurement is eval/separation.py against a sealed set, not
    # a loss on a test split.
    dump("test.jsonl", val_rows)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


def build_plan(config: dict, config_path: Path) -> dict:
    peft = config.get("peft", {})
    trainer = config.get("trainer", {})
    optim = config.get("optim", {})
    data = config.get("data", {})
    output = config.get("output", {})
    base = config.get("base_model", {})

    hf_modules = list(peft.get("target_modules", []))
    return {
        "tenant": config.get("tenant"),
        "config_path": str(config_path),
        "hf_id": base.get("hf_id"),
        "nemo_config_class": base.get("nemo_config_class"),
        "nemo_model_class": base.get("nemo_model_class"),
        "dim": peft.get("dim"),
        "alpha": peft.get("alpha"),
        "dropout": peft.get("dropout"),
        "hf_target_modules": hf_modules,
        "nemo_target_modules": translate_target_modules(hf_modules),
        "use_canonical": bool(peft.get("nemo_use_canonical", False)),
        "seed": trainer.get("seed"),
        "max_epochs": trainer.get("max_epochs"),
        "precision": trainer.get("precision"),
        "devices": trainer.get("devices", 1),
        "num_nodes": trainer.get("num_nodes", 1),
        "lr": optim.get("lr"),
        "min_lr": optim.get("min_lr"),
        "scheduler": optim.get("scheduler"),
        "warmup_steps": optim.get("warmup_steps"),
        "weight_decay": optim.get("weight_decay"),
        "use_distributed_optimizer": optim.get("use_distributed_optimizer", True),
        "train_file": data.get("train_file"),
        "nemo_data_root": data.get("nemo_data_root"),
        "validation_fraction": data.get("validation_fraction", 0.1),
        "seq_length": data.get("seq_length"),
        "global_batch_size": data.get("global_batch_size"),
        "micro_batch_size": data.get("micro_batch_size", 1),
        "nemo_ckpt_dir": output.get("nemo_ckpt_dir"),
        "hf_adapter_dir": output.get("hf_adapter_dir"),
        "log_dir": output.get("log_dir", "train/logs"),
        "experiment_name": output.get("experiment_name"),
    }


def print_plan(plan: dict, stream=None) -> None:
    stream = stream or sys.stdout
    write = lambda line: print(line, file=stream)  # noqa: E731

    write("")
    write("RESOLVED TRAINING PLAN")
    write(f"  config            {plan['config_path']}")
    write(f"  tenant            {plan['tenant']}")
    write(f"  base model        {plan['hf_id']}")
    write("")
    write("  LoRA")
    write(f"    dim (rank)      {plan['dim']}          # NeMo calls rank `dim`, not `r`")
    write(f"    alpha           {plan['alpha']}")
    write(f"    dropout         {plan['dropout']}")
    write(f"    HF modules      {', '.join(plan['hf_target_modules'])}")
    write(f"    NeMo modules    {', '.join(plan['nemo_target_modules'])}")
    write("                    (Megatron fuses q/k/v into linear_qkv and")
    write("                     gate/up into linear_fc1; that is why 7 -> 4)")
    write(f"    canonical       {plan['use_canonical']}"
          "        # true = one adapter per HF module, matching HF exactly")
    write("")
    write("  trainer")
    write(f"    seed            {plan['seed']}")
    write(f"    max_epochs      {plan['max_epochs']}")
    write(f"    precision       {plan['precision']}")
    write(f"    devices         {plan['devices']} x {plan['num_nodes']} node(s)")
    write("")
    write("  optim")
    write(f"    lr              {plan['lr']}  ({plan['scheduler']}, min {plan['min_lr']},"
          f" warmup {plan['warmup_steps']})")
    write(f"    weight_decay    {plan['weight_decay']}")
    write("")
    write("  data")
    write(f"    chat jsonl      {plan['train_file']}")
    write(f"    nemo root       {plan['nemo_data_root']}")
    write(f"    seq_length      {plan['seq_length']}")
    write(f"    global batch    {plan['global_batch_size']}"
          f"  (micro {plan['micro_batch_size']}, so grad accum "
          f"{plan['global_batch_size'] // max(1, plan['micro_batch_size'] * plan['devices'])})")
    write("")
    write("  output")
    write(f"    nemo ckpt       {plan['nemo_ckpt_dir']}")
    write(f"    hf adapter      {plan['hf_adapter_dir']}  (via train/convert_to_hf.py)")
    write(f"    logs            {plan['log_dir']}")
    write("")


def print_nemo_calls(plan: dict, stream=None) -> None:
    """The exact NeMo API calls the real run makes. Printed so a reviewer can
    check them against the docs without reading the rest of this file."""
    stream = stream or sys.stdout
    modules = plan["nemo_target_modules"]
    lora_cls = "llm.peft.CanonicalLoRA" if plan["use_canonical"] else "llm.peft.LoRA"
    print(
        f"""
NEMO CALLS THIS PLAN WOULD MAKE

  from nemo.collections import llm
  from nemo import lightning as nl
  from megatron.core.optimizer import OptimizerConfig

  # 1. import the HF checkpoint into NeMo format (once, cached)
  llm.import_ckpt(
      model=llm.{plan['nemo_model_class']}(
          config=llm.{plan['nemo_config_class']}()),   # CHECK: class names
      source='hf://{plan['hf_id']}',
  )

  # 2. LoRA. Rank is `dim`.
  peft = {lora_cls}(
      dim={plan['dim']},
      alpha={plan['alpha']},
      dropout={plan['dropout']},
      target_modules={modules!r},
  )

  # 3. data
  data = llm.FineTuningDataModule(
      dataset_root='{plan['nemo_data_root']}',
      seq_length={plan['seq_length']},
      micro_batch_size={plan['micro_batch_size']},
      global_batch_size={plan['global_batch_size']},
  )

  # 4. trainer
  trainer = nl.Trainer(
      devices={plan['devices']},
      num_nodes={plan['num_nodes']},
      accelerator='gpu',
      max_epochs={plan['max_epochs']},
      strategy=nl.MegatronStrategy(),
      plugins=nl.MegatronMixedPrecision(precision='{plan['precision']}'),
  )

  # 5. optimizer + cosine schedule
  optim = nl.MegatronOptimizerModule(
      config=OptimizerConfig(
          optimizer='adam',
          lr={plan['lr']},
          weight_decay={plan['weight_decay']},
          bf16=True,
          use_distributed_optimizer={plan['use_distributed_optimizer']},
      ),
      lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
          warmup_steps={plan['warmup_steps']},
          min_lr={plan['min_lr']},
      ),
  )

  # 6. go
  llm.finetune(model=model, data=data, trainer=trainer, peft=peft,
               optim=optim, log=nl.NeMoLogger(name='{plan['experiment_name']}'))
""",
        file=stream,
    )


# --------------------------------------------------------------------------
# the real run
# --------------------------------------------------------------------------


def run_training(plan: dict) -> int:
    """Import NeMo and run. Nothing above this function imports NeMo."""
    try:
        from nemo.collections import llm  # noqa: PLC0415
        from nemo import lightning as nl  # noqa: PLC0415
        from megatron.core.optimizer import OptimizerConfig  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"train_lora: NeMo is not importable in this environment ({exc}).\n"
            f"            This script runs inside nvcr.io/nvidia/nemo, not on a\n"
            f"            laptop. Use --dry-run here, or take the documented\n"
            f"            fallback route: train/train_lora_hf.py.\n"
            f"            See train/README.md for when the fallback triggers.",
            file=sys.stderr,
        )
        return 2

    # CHECK: seed pinning. NeMo 2.x does not document its own seed argument on
    # CHECK: llm.finetune or nl.Trainer. Lightning's seed_everything is the
    # CHECK: documented mechanism for the underlying framework and is what is
    # CHECK: used here, but whether it reaches Megatron's data sampler and
    # CHECK: parallel RNG state has NOT been confirmed. Until it is, treat two
    # CHECK: runs at the same seed as reproducible-ish, not bit-identical, and
    # CHECK: say so wherever a number from this training run is reported.
    try:
        from lightning.pytorch import seed_everything  # noqa: PLC0415
    except ImportError:
        try:
            from pytorch_lightning import seed_everything  # noqa: PLC0415
        except ImportError:
            seed_everything = None
    if seed_everything is not None:
        seed_everything(plan["seed"], workers=True)
        print(f"train_lora: seeded everything with {plan['seed']}")
    else:
        print("train_lora: WARNING could not import seed_everything; run is NOT seeded")

    # CHECK: the NeMo config/model class names for this base model. The Llama
    # CHECK: Nemotron page documents the family but publishes an import_ckpt
    # CHECK: example only for the Ultra 253B variant
    # CHECK: (https://docs.nvidia.com/nemo-framework/user-guide/25.09/llms/llama_nemotron.html).
    # CHECK: The Nano 8B names in config_*.yaml are a guess. Resolve them from
    # CHECK: `dir(llm)` inside the container. This block refuses to guess a
    # CHECK: substitute: three epochs on the wrong architecture is expensive and
    # CHECK: fails silently.
    config_cls = getattr(llm, plan["nemo_config_class"], None)
    model_cls = getattr(llm, plan["nemo_model_class"], None)
    if config_cls is None or model_cls is None:
        candidates = sorted(
            name for name in dir(llm)
            if "Nemotron" in name or ("Llama31" in name and "Config" in name)
        )
        print(
            f"train_lora: cannot resolve the NeMo classes named in the config.\n"
            f"            nemo_config_class = {plan['nemo_config_class']!r} -> "
            f"{'found' if config_cls else 'NOT FOUND'}\n"
            f"            nemo_model_class  = {plan['nemo_model_class']!r} -> "
            f"{'found' if model_cls else 'NOT FOUND'}\n"
            f"            Candidates visible in this NeMo build:\n"
            + "".join(f"              {name}\n" for name in candidates)
            + f"            Fix base_model.* in {plan['config_path']} and note the\n"
            f"            correct names in change_log.md. Refusing to guess.",
            file=sys.stderr,
        )
        return 2

    model = model_cls(config=config_cls())

    print(f"train_lora: importing {plan['hf_id']} into NeMo format (cached after the first run)")
    llm.import_ckpt(model=model, source=f"hf://{plan['hf_id']}")

    lora_cls = llm.peft.CanonicalLoRA if plan["use_canonical"] else llm.peft.LoRA
    peft = lora_cls(
        dim=plan["dim"],              # rank. NeMo spells it `dim`.
        alpha=plan["alpha"],
        dropout=plan["dropout"],
        target_modules=plan["nemo_target_modules"],
    )

    data = llm.FineTuningDataModule(
        dataset_root=str(resolve_path(plan["nemo_data_root"])),
        seq_length=plan["seq_length"],
        micro_batch_size=plan["micro_batch_size"],
        global_batch_size=plan["global_batch_size"],
    )

    trainer = nl.Trainer(
        devices=plan["devices"],
        num_nodes=plan["num_nodes"],
        accelerator="gpu",
        max_epochs=plan["max_epochs"],
        strategy=nl.MegatronStrategy(),
        plugins=nl.MegatronMixedPrecision(precision=plan["precision"]),
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=plan["lr"],
            weight_decay=plan["weight_decay"],
            bf16=True,
            use_distributed_optimizer=plan["use_distributed_optimizer"],
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=plan["warmup_steps"],
            min_lr=plan["min_lr"],
        ),
    )

    log_dir = resolve_path(plan["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    print("train_lora: starting llm.finetune")
    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        peft=peft,
        optim=optim,
        log=nl.NeMoLogger(name=plan["experiment_name"], log_dir=str(log_dir)),
    )
    print("train_lora: finetune returned")
    print(f"train_lora: convert the checkpoint with:\n"
          f"  python train/convert_to_hf.py --tenant {plan['tenant']}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class Tee:
    """Write to stdout and to the run log at the same time."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_lora.py",
        description="LoRA fine-tune one tenant with the NeMo Framework 2.x API.",
    )
    parser.add_argument("--tenant", choices=TENANTS, required=True, help="which tenant to train")
    parser.add_argument("--config", default=None, help="override the config path")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve the config, convert the data, print the plan; never import NeMo",
    )
    parser.add_argument(
        "--prepare-data-only", action="store_true",
        help="write the NeMo dataset_root and exit",
    )
    parser.add_argument(
        "--force-data", action="store_true",
        help="rewrite the NeMo dataset_root even if it already exists",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else ROOT / "train" / f"config_{args.tenant}.yaml"
    config = load_config(config_path)

    if config.get("tenant") != args.tenant:
        print(
            f"train_lora: --tenant is {args.tenant!r} but {config_path} says "
            f"{config.get('tenant')!r}. Refusing to run: this is exactly how one "
            f"tenant's data ends up in the other tenant's adapter.",
            file=sys.stderr,
        )
        return 2

    plan = build_plan(config, config_path)

    log_dir = resolve_path(plan["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log = log_dir / f"train_{args.tenant}_{stamp}.log"

    handle = run_log.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, handle)
    try:
        print(f"train_lora: run log {run_log}")
        print_plan(plan)

        # ---- data ----
        train_file = resolve_path(plan["train_file"])
        dataset_root = resolve_path(plan["nemo_data_root"])
        if not train_file.exists():
            print(
                f"train_lora: no training data at {train_file}.\n"
                f"            Run: python data/generate.py package\n"
                f"            (Stage 1 is blocked on ANTHROPIC_API_KEY; until it "
                f"runs, there is no training data and nothing to train on.)",
                file=sys.stderr,
            )
            return 2

        chat_rows = read_chat_jsonl(train_file)
        if not chat_rows:
            print(f"train_lora: {train_file} has no chat rows", file=sys.stderr)
            return 2

        if dataset_root.exists() and not args.force_data:
            print(f"train_lora: dataset_root {dataset_root} exists; leaving it "
                  f"(pass --force-data to rewrite)")
        else:
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            train_rows, val_rows = convert_to_nemo_jsonl(
                chat_rows, plan["validation_fraction"], plan["seed"]
            )
            write_nemo_dataset(dataset_root, train_rows, val_rows)
            print(f"train_lora: {len(chat_rows)} chat rows -> {len(train_rows)} train / "
                  f"{len(val_rows)} validation, written to {dataset_root}")
            print("train_lora: NOTE the chat rows were flattened to NeMo's "
                  "{input, output} shape;")
            print("train_lora:      the system message was prepended to the input, "
                  "not dropped.")

        if args.prepare_data_only:
            print("train_lora: --prepare-data-only, stopping here")
            return 0

        if args.dry_run:
            print_nemo_calls(plan)
            print("train_lora: --dry-run, NeMo was never imported. Nothing trained.")
            return 0

        return run_training(plan)
    finally:
        sys.stdout = original_stdout
        handle.close()
        print(f"train_lora: log written to {run_log}")


if __name__ == "__main__":
    sys.exit(main())
