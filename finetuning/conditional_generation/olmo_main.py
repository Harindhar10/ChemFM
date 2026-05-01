import argparse
import os
import sys
import random
import numpy as np
import torch
import transformers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils import (
    ModelArguments,
    DataArguments,
    smart_tokenizer_and_embedding_resize,
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
    eval_steps: int = field(default=500)
    save_steps: int = field(default=500)
    precision: str = field(default="bf16-mixed")
    training_args_file: Optional[str] = field(default=None)
    num_workers: int = field(default=4)


def load_model_and_tokenizer(model_args, args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        # bfloat16 for training stability (float16 is for inference only)
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Do NOT pass device_map="auto" — Lightning manages device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrain_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map=None,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # lora_rank, lora_alpha_ratio, lora_dropout, lora_target_modules all come from ModelArguments
    target_modules = (
        [m.strip() for m in args.lora_target_modules.split(",")]
        if args.lora_target_modules
        else ["q_proj", "k_proj", "v_proj"]
    )
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=int(args.lora_alpha_ratio * args.lora_rank),
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    if tokenizer.pad_token is None:
        special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    # Required for gradient checkpointing compatibility
    model.config.use_cache = False

    model.print_trainable_parameters()
    return model, tokenizer


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

    model, tokenizer = load_model_and_tokenizer(model_args, args)

    data_module = make_data_module(tokenizer=tokenizer, ignore_index=IGNORE_INDEX, args=args)

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

    lit_model = OlmoConditionalGenModule(model=model, tokenizer=tokenizer, args=args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="olmo-cond-gen-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=args.save_steps,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_logger = CSVLogger(save_dir=args.output_dir, name="logs")

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=args.logging_steps,
        # Integer: check validation every N batches (not optimizer steps)
        val_check_interval=args.eval_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=csv_logger,
        enable_progress_bar=True,
        # BnB custom CUDA kernels are non-deterministic; forcing determinism raises errors
        deterministic=False,
        # DDP is compatible with bitsandbytes; FSDP and DeepSpeed are not
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    lit_model.save_adapter(final_adapter_dir)


if __name__ == "__main__":
    main()
