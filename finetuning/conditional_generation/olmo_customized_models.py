import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from typing import List, Any, Dict


class OlmoConditionalGenModule(pl.LightningModule):

    def __init__(self, model, tokenizer, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.numerical_embedding = nn.Linear(1, model.config.hidden_size, bias=True)
        self.save_hyperparameters(ignore=["model", "tokenizer"])

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
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
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

    def save_adapter(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        torch.save(
            self.numerical_embedding.state_dict(),
            os.path.join(output_dir, "numerical_embedding.pt"),
        )
        print(f"Adapter and numerical embedding saved to {output_dir}")
