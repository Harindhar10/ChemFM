import argparse
import os
import sys
import importlib
import numpy as np
import torch
import transformers
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from rdkit import RDLogger, Chem
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from utils import (
    ModelArguments,
    DataArguments,
    make_test_data_module,
)
from metric_calculator import get_similarity, get_scaffold
from olmo_customized_models import OlmoConditionalGenModule

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

RDLogger.DisableLog("rdApp.*")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class OlmoEvalArguments:
    output_dir: str = field(default="./outputs/olmo")
    seed: int = field(default=0)
    training_args_file: Optional[str] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)


def load_model_and_tokenizer(model_args, args):
    model_path = model_args.model_path  # HF repo ID or local path to fine-tuned model
    if not model_path:
        raise ValueError(
            "model_path is not set. Pass --model_path <hf-repo-id> or "
            "--model_path <path/to/final_model> (the directory saved after training)."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lit_model = OlmoConditionalGenModule(model_id=model_path, tokenizer=tokenizer, args=args)
    lit_model.configure_model()
    lit_model.model.config.use_cache = True

    num_emb_local = os.path.join(model_path, "numerical_embedding.pt")
    if os.path.exists(num_emb_local):
        num_emb_path = num_emb_local
    else:
        num_emb_path = hf_hub_download(repo_id=model_path, filename="numerical_embedding.pt")

    state_dict = torch.load(num_emb_path, map_location="cpu", weights_only=True)
    lit_model.numerical_embedding.load_state_dict(state_dict)

    lit_model.eval()
    return lit_model, tokenizer


def generate(model, loader, accelerator, tokenizer, max_length):
    model.eval()
    inner_model = accelerator.unwrap_model(model).model

    df = []
    pbar = tqdm(loader, desc="Evaluating...", leave=False)
    for it, batch in enumerate(pbar):
        sub_df = dict()

        assert batch["input_ids"].shape[0] == 1, "The batch size should be 1"

        temperature = batch["temperature"][0]
        property_names = batch["property_names"][0]
        non_normalized_properties = batch["non_normalized_properties"][0]
        scaffold = batch["scaffold"][0]
        steps = max_length - batch["input_ids"].shape[1]

        with torch.no_grad():
            # Inject numerical embeddings once for the prompt; generate() handles the rest.
            inputs_embeds = model.inject_numerical_embeddings(
                batch["input_ids"],
                batch["properties"],
                batch["properties_index"],
            )
            output_ids = inner_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                max_new_tokens=steps,
                do_sample=True,
                temperature=float(temperature),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generations = [g.replace(" ", "") for g in generations]

        predictions = []
        for generation in generations:
            try:
                predictions.append(Chem.MolToSmiles(Chem.MolFromSmiles(generation)))
            except Exception:
                predictions.append("")

        sub_df["SMILES"] = predictions[0]
        sub_df["property_names"] = property_names
        sub_df["property"] = batch["properties"][0]
        sub_df["non_normalized_properties"] = non_normalized_properties
        if scaffold is not None:
            sub_df["scaffold"] = scaffold

        gathered_sub_df = accelerator.gather_for_metrics([sub_df])
        df.extend(gathered_sub_df)

    df = pd.DataFrame(df)
    return df


def phrase_df(df):
    metric_calculator = importlib.import_module("metric_calculator")

    new_df = []
    for i in range(len(df)):
        sub_df = dict()

        smiles = df.iloc[i]["SMILES"]
        property_names = df.iloc[i]["property_names"]
        non_normalized_properties = df.iloc[i]["non_normalized_properties"]

        sub_df["SMILES"] = smiles

        if "scaffold" in df.columns:
            scaffold = df.iloc[i]["scaffold"]
            sub_df["scaffold"] = scaffold
            if smiles == "":
                sub_df["Similarity"] = np.nan
            else:
                sub_df["Similarity"] = get_similarity(get_scaffold(smiles), scaffold)

        for j in range(len(property_names)):
            property_name = property_names[j]
            non_normalized_property = non_normalized_properties[j]

            sub_df[f"{property_name}_condition"] = non_normalized_property

            if smiles == "":
                sub_df[f"{property_name}_measured"] = np.nan
            else:
                property_eval_func = getattr(
                    metric_calculator, f"compute_{property_name}"
                )
                sub_df[f"{property_name}_measured"] = property_eval_func(
                    Chem.MolFromSmiles(smiles)
                )

        new_df.append(sub_df)

    return pd.DataFrame(new_df)


def evaluate():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, OlmoEvalArguments)
    )
    model_args, data_args, eval_args, _ = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if eval_args.training_args_file is not None:
        explicit_keys = {
            arg.lstrip("-").split("=")[0].replace("-", "_")
            for arg in sys.argv if arg.startswith("--")
        }
        pre_yaml = {**vars(model_args), **vars(data_args), **vars(eval_args)}
        cli_overrides = {k: v for k, v in pre_yaml.items() if k in explicit_keys}
        model_args, data_args, eval_args = hfparser.parse_yaml_file(
            eval_args.training_args_file
        )
        for k, v in cli_overrides.items():
            for ns in (model_args, data_args, eval_args):
                if hasattr(ns, k):
                    setattr(ns, k, v)

    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(eval_args))

    model, tokenizer = load_model_and_tokenizer(model_args, args)

    accelerator = Accelerator()

    data_module = make_test_data_module(
        tokenizer=tokenizer, ignore_index=IGNORE_INDEX, args=args
    )
    data_collator = data_module["data_collator"]
    if args.max_test_samples is not None:
        data_module["test_dataset"] = data_module["test_dataset"].select(range(args.max_test_samples))
    test_loader = DataLoader(
        data_module["test_dataset"],
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    df = generate(
        model,
        test_loader,
        accelerator,
        tokenizer,
        args.source_max_len + args.target_max_len,
    )

    if accelerator.is_main_process:
        df = phrase_df(df)
        output_path = args.generation_output_path
        folder_name = os.path.dirname(output_path)
        if folder_name and not os.path.exists(folder_name):
            os.makedirs(folder_name)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    evaluate()
