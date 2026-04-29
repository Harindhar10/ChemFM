import argparse
import transformers
import torch

from utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    smart_tokenizer_and_embedding_resize,
    make_data_module
)
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoConfig,
    Seq2SeqTrainer
)
import numpy as np
import random
import os
from llama_customized_models import LlamaForCausalLMWithNumericalEmbedding

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from rdkit import RDLogger
# Suppress RDKit INFO messages
RDLogger.DisableLog('rdApp.*')

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def load_model_and_tokenizer(model_args, args):

    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    # load tokenizer if it is provided
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=args.trust_remote_code
        )

    # load model
    config = AutoConfig.from_pretrained(
        model_args.pretrain_model,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )

    model = LlamaForCausalLMWithNumericalEmbedding.from_pretrained(
        model_args.pretrain_model,
        config=config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    
    # the finetune tokenizer could be in different size with pretrain tokenizer, and also, we need to add PAD_TOKEN
    special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # apply lora if specified
    if args.lora:
        raise NotImplementedError("LORA is not implemented yet.")
    
    try: 
        model.print_trainable_parameters()
    except:
        dtypes = {}
        dtypes_trainable = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes: 
                dtypes[dtype] = 0
                dtypes_trainable[dtype] = 0
            dtypes[dtype] += p.numel()
            if p.requires_grad:
                dtypes_trainable[dtype] += p.numel()

        total = 0
        trainable = 0
        for k, v in dtypes.items(): total+= v
        for k, v in dtypes_trainable.items(): trainable += v
        for k, v in dtypes.items():
            print(f"dtypes: {k} || trainable params: {dtypes_trainable[k]} || {k} params: {v} || all params: {total} || trainable%: {dtypes_trainable[k]/v:.3f}")
    
    model.config.use_cache = False

    return model, tokenizer


def main():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    import sys
    model_args, data_args, training_args, _ = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    # if training_args_file is provided, overwrite the arguments with the arguments in the file,
    # then re-apply values that were explicitly passed on the CLI so they take precedence
    if training_args.training_args_file is not None:
        explicit_keys = {
            arg.lstrip("-").split("=")[0].replace("-", "_")
            for arg in sys.argv if arg.startswith("--")
        }
        pre_yaml = {**vars(model_args), **vars(data_args), **vars(training_args)}
        cli_overrides = {k: v for k, v in pre_yaml.items() if k in explicit_keys}
        model_args, data_args, training_args = hfparser.parse_yaml_file(training_args.training_args_file)
        for k, v in cli_overrides.items():
            for ns in (model_args, data_args, training_args):
                if hasattr(ns, k):
                    setattr(ns, k, v)

    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # set seed for transformers, torch, and numpy, and random
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, args)

    data_module = make_data_module(tokenizer=tokenizer, ignore_index=IGNORE_INDEX, 
                                   args=args)
    
    # set the min learning rate for the optimizer
    if args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = args.min_learning_rate

    # set the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()