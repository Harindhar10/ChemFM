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
from peft import PeftModel
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from rdkit import RDLogger, Chem
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils import (
    ModelArguments,
    DataArguments,
    smart_tokenizer_and_embedding_resize,
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
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")


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
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrain_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(base_model, model_args.model_path)

    if tokenizer.pad_token is None:
        special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    # Enable KV cache for faster autoregressive generation
    model.config.use_cache = True

    lit_model = OlmoConditionalGenModule(model=model, tokenizer=tokenizer, args=args)

    numerical_embedding_path = os.path.join(
        model_args.model_path, "numerical_embedding.pt"
    )
    state_dict = torch.load(numerical_embedding_path, map_location="cpu")
    lit_model.numerical_embedding.load_state_dict(state_dict)

    lit_model.eval()
    return lit_model, tokenizer


def generate(model, loader, accelerator, tokenizer, max_length):
    model.eval()

    df = []
    pbar = tqdm(loader, desc="Evaluating...", leave=False)
    for it, batch in enumerate(pbar):
        sub_df = dict()

        batch_size = batch["input_ids"].shape[0]
        assert batch_size == 1, "The batch size should be 1"

        temperature = batch["temperature"][0]
        property_names = batch["property_names"][0]
        non_normalized_properties = batch["non_normalized_properties"][0]
        scaffold = batch["scaffold"][0]

        # Keep properties and properties_index in batch for embedding injection each step.
        # Remove the non-tensor meta fields before the generation loop.
        del batch["temperature"]
        del batch["property_names"]
        del batch["non_normalized_properties"]
        del batch["scaffold"]

        input_length = batch["input_ids"].shape[1]
        steps = max_length - input_length

        with torch.no_grad():
            early_stop_flags = torch.zeros(1, dtype=torch.bool).to(
                batch["input_ids"].device
            )
            for k in range(steps):
                # Inject numerical embeddings at the fixed prefix positions each step.
                # properties_index values are absolute positions and never shift
                # as new tokens are appended to the right.
                inputs_embeds = model.inject_numerical_embeddings(
                    batch["input_ids"],
                    batch["properties"],
                    batch["properties_index"],
                )
                outputs = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1)

                ix[early_stop_flags] = tokenizer.eos_token_id

                batch["input_ids"] = torch.cat([batch["input_ids"], ix], dim=-1)
                # Extend attention mask to cover the newly appended token
                batch["attention_mask"] = torch.cat(
                    [
                        batch["attention_mask"],
                        torch.ones(
                            (1, 1),
                            dtype=batch["attention_mask"].dtype,
                            device=batch["attention_mask"].device,
                        ),
                    ],
                    dim=-1,
                )
                early_stop_flags |= ix.squeeze() == tokenizer.eos_token_id

                if torch.all(early_stop_flags):
                    break

        generations = tokenizer.batch_decode(
            batch["input_ids"][:, input_length:], skip_special_tokens=True
        )
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
        model_args, data_args, eval_args = hfparser.parse_yaml_file(
            eval_args.training_args_file
        )

    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(eval_args))

    model, tokenizer = load_model_and_tokenizer(model_args, args)

    accelerator = Accelerator()

    data_module = make_test_data_module(
        tokenizer=tokenizer, ignore_index=IGNORE_INDEX, args=args
    )
    data_collator = data_module["data_collator"]
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
