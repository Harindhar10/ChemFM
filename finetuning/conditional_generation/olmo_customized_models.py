import os
from functools import partial

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoConfig, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import List, Any, Dict


class OlmoConditionalGenModule(pl.LightningModule):

    def __init__(self, model_id, tokenizer, args):
        super().__init__()
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.args = args
        self.model = None

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.numerical_embedding = nn.Linear(1, config.hidden_size, bias=True)
        self.save_hyperparameters(ignore=["tokenizer"])

    def configure_model(self):
        if self.model is not None:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            device_map=None,
            attn_implementation="flash_attention_2",
        )
        self.numerical_embedding = self.numerical_embedding.to(torch.bfloat16)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Move tensors to device but leave Python lists (properties, properties_index) as-is.
        # Lightning's default hook would crash trying to call .to() on nested lists.
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def inject_numerical_embeddings(
        self,
        input_ids: torch.LongTensor,
        properties: List[List[float]],
        properties_index: List[List[int]],
    ) -> torch.FloatTensor:
        # .clone() is required: with gradient checkpointing the embedding output is a
        # leaf variable, and in-place index assignment on a leaf that requires grad raises
        # "RuntimeError: a view of a leaf Variable that requires grad is being used in an
        # in-place operation."
        embeddings = self.model.get_input_embeddings()(input_ids).clone()
        embed_dtype = embeddings.dtype
        embed_device = embeddings.device

        for i, (props, props_index) in enumerate(zip(properties, properties_index)):
            if len(props) == 0:
                continue
            props_tensor = torch.tensor(
                props, device=embed_device, dtype=torch.float32
            ).unsqueeze(1)  # (N_props, 1)
            num_embeds = self.numerical_embedding(props_tensor)  # (N_props, H) float32
            num_embeds = num_embeds.to(dtype=embed_dtype)        # cast to bfloat16
            embeddings[i, props_index, :] = num_embeds

        return embeddings

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        properties = batch["properties"]
        properties_index = batch["properties_index"]

        inputs_embeds = self.inject_numerical_embeddings(input_ids, properties, properties_index)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        properties = batch["properties"]
        properties_index = batch["properties_index"]

        inputs_embeds = self.inject_numerical_embeddings(input_ids, properties, properties_index)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.log("val_loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "bias" in name or "layer_norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.args.warmup_ratio * total_steps)

        if self.args.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        clip_val = gradient_clip_val if gradient_clip_val is not None else 1.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        torch.save(
            self.numerical_embedding.state_dict(),
            os.path.join(output_dir, "numerical_embedding.pt"),
        )
        print(f"Model and numerical embedding saved to {output_dir}")
