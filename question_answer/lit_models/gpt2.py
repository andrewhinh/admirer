import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Tuple
import wandb


import question_answer.metadata.pica as metadata


OPTIMIZER = "Adam"
LR = 1e-4
ONE_CYCLE_TOTAL_STEPS = 100

TOP_K = 1000
TOP_P = 0.95
MAX_LABEL_LENGTH = metadata.MAX_LABEL_LENGTH
LABEL_MASK = -100


class GPT2(pl.LightningModule):
    """
    GPT2 PyTorch-Lightning class.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        args: argparse.Namespace = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.top_k = self.args.get("top_k", TOP_K)
        self.top_p = self.args.get("top_p", TOP_P)
        self.max_label_length = self.args.get("max_label_length", MAX_LABEL_LENGTH)
        self.label_mask = self.args.get("label_mask", LABEL_MASK)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # only allow training of cross attention parameters
        for layer in model.decoder.transformer.h:
            layer.crossattention.train()
            for p in layer.crossattention.parameters():
                p.requires_grad = True
            layer.ln_cross_attn.train()
            for p in layer.ln_cross_attn.parameters():
                p.requires_grad = True

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)

        parser.add_argument("--top_k", type=int, default=TOP_K)
        parser.add_argument("--top_p", type=int, default=TOP_P)
        parser.add_argument("--max_label_length", type=int, default=MAX_LABEL_LENGTH)
        parser.add_argument("--label_mask", type=float, default=LABEL_MASK)
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation/loss"}

    def common_step(self, batch: Tuple[torch.FloatTensor, List[str]]) -> torch.FloatTensor:
        images, captions = batch
        tokenized_captions = {
            k: v.to(self.device)
            for k, v in self.tokenizer(
                captions,
                max_length=self.max_label_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).items()
        }
        labels = tokenized_captions["input_ids"].clone()
        labels[tokenized_captions["attention_mask"] == 0] = self.label_mask
        encoder_outputs = self.model.encoder(pixel_values=images)
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=tokenized_captions["input_ids"],
            decoder_attention_mask=tokenized_captions["attention_mask"],
            labels=labels,
            return_dict=True,
        )

        return outputs["loss"]

    def training_step(self, batch: Tuple[torch.FloatTensor, List[str]], batch_idx: int) -> torch.FloatTensor:
        loss = self.common_step(batch)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Tuple[torch.FloatTensor, List[str]], batch_idx: int):
        loss = self.common_step(batch)
        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Tuple[torch.FloatTensor, List[str]], batch_idx: int):
        loss = self.common_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_after_backward(self):
        if self.trainer.global_step % 50 == 0:  # don't make the tf file huge
            for name, param in self.model.named_parameters():
                if "weight" in name and "norm" not in name and param.requires_grad:
                    self.logger.experiment.log({f"{name}_grad": wandb.Histogram(param.grad.detach().cpu())})
                    self.logger.experiment.log({f"{name}": wandb.Histogram(param.detach().cpu())})
