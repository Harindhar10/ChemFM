import argparse
import os
import sys
import random
from functools import partial

import numpy as np
import torch
import transformers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from huggingface_hub import HfApi

from utils import (
    ModelArguments,
    DataArguments,
    make_data_module,
)
from olmo_customized_models import OlmoConditionalGenModule

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class OlmoTrainingArguments:
    output_dir: str = field(default="./outputs/olmo")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=2e-4)
    min_learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)
    seed: int = field(default=0)
    logging_steps: int = field(default=10)
    num_val_per_epoch: int = field(default=1)
    save_steps: int = field(default=500)
    precision: str = field(default="bf16-mixed")
    training_args_file: Optional[str] = field(default=None)
    run_id: Optional[str] = field(default=None)
    num_workers: int = field(default=4)
    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    wandb_logging: bool = field(default=False)
    wandb_key: Optional[str] = field(default=None)
    wandb_notes: Optional[str] = field(default=None)
    save_name: Optional[str] = field(default=None)


def load_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, OlmoTrainingArguments)
    )
    model_args, data_args, training_args, _ = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    # YAML config file support: load YAML, then re-apply CLI overrides
    if training_args.training_args_file is not None:
        explicit_keys = {
            arg.lstrip("-").split("=")[0].replace("-", "_")
            for arg in sys.argv if arg.startswith("--")
        }
        pre_yaml = {**vars(model_args), **vars(data_args), **vars(training_args)}
        cli_overrides = {k: v for k, v in pre_yaml.items() if k in explicit_keys}
        model_args, data_args, training_args = hfparser.parse_yaml_file(
            training_args.training_args_file
        )
        for k, v in cli_overrides.items():
            for ns in (model_args, data_args, training_args):
                if hasattr(ns, k):
                    setattr(ns, k, v)

    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    tokenizer = load_tokenizer(model_args)

    data_module = make_data_module(tokenizer=tokenizer, ignore_index=IGNORE_INDEX, args=args)

    if args.max_train_samples is not None:
        data_module["train_dataset"] = data_module["train_dataset"].select(range(args.max_train_samples))
    if args.max_val_samples is not None:
        data_module["eval_dataset"] = data_module["eval_dataset"].select(range(args.max_val_samples))

    train_loader = DataLoader(
        data_module["train_dataset"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_module["data_collator"],
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        data_module["eval_dataset"],
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_module["data_collator"],
        num_workers=args.num_workers,
        pin_memory=True,
    )

    lit_model = OlmoConditionalGenModule(
        model_id=args.pretrain_model, tokenizer=tokenizer, args=args
    )

    # Force model init to get transformer layer class for FSDP wrapping policy
    lit_model.configure_model()

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(lit_model.model.model.layers[0])},
    )
    fsdp_strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy=auto_wrap_policy,
        cpu_offload=False,
        use_orig_params=True,
        sync_module_states=True,
        state_dict_type="full",
    )

    run_name = args.run_id or "olmo-cond-gen"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=run_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    os.makedirs(args.output_dir, exist_ok=True)
    loggers = [CSVLogger(save_dir=args.output_dir, name="logs")]

    if args.wandb_logging:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        loggers.append(WandbLogger(
            project="olmo-conditional-generation",
            name=run_name,
            log_model=False,
            notes=args.wandb_notes,
        ))

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=fsdp_strategy,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=args.logging_steps,
        val_check_interval=1.0 / args.num_val_per_epoch,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=loggers,
        enable_progress_bar=True,
        deterministic=False,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.strategy.barrier()
    if trainer.global_rank == 0:
        final_model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)

        ckpt_path = os.path.join(args.output_dir, f"{run_name}.ckpt")
        print(f"Checkpoint path: {ckpt_path}")

        reload_model = AutoModelForCausalLM.from_pretrained(
            args.pretrain_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        num_emb_state = {k[len("numerical_embedding."):]: v for k, v in state.items() if k.startswith("numerical_embedding.")}

        reload_model.load_state_dict(model_state, strict=True)

        if args.save_name:
            print(f"Pushing model to HF Hub: {args.save_name}")
            reload_model.push_to_hub(args.save_name)
            tokenizer.push_to_hub(args.save_name)
            num_emb_path = os.path.join(args.output_dir, "numerical_embedding.pt")
            torch.save(num_emb_state, num_emb_path)
            HfApi().upload_file(
                path_or_fileobj=num_emb_path,
                path_in_repo="numerical_embedding.pt",
                repo_id=args.save_name,
            )
            print(f"Model pushed to {args.save_name}")
        else:
            reload_model.save_pretrained(final_model_dir)
            torch.save(num_emb_state, os.path.join(final_model_dir, "numerical_embedding.pt"))
            tokenizer.save_pretrained(final_model_dir)
            print(f"Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()
