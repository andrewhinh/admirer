# Libraries
from pathlib import Path

import data
import model as md
import torch
from torch.utils.data import DataLoader, SequentialSampler
import transformers
import wandb


# Variables
HOME_DIR = Path("/home/andrewhinh/Desktop/Projects/")
CURR_DIR = Path("./")
PRETRAINED_MODEL_PATH = "large_both_knowledge/"
STAGED_MODEL_FILENAME = "model.pt"
TO_PROJECT = "admirer-training"
LOG_DIR = Path("training") / "logs"
STAGED_MODEL_TYPE = "prod-ready"

from_model_path = HOME_DIR / PRETRAINED_MODEL_PATH
to_model_path = CURR_DIR / STAGED_MODEL_FILENAME

text_maxlength = 64
n_context = 40
per_gpu_batch_size = 1
num_workers = 16


# Load best pre-trained model from https://github.com/guilk/KAT
model_class = md.FiDT5
model = model_class.from_pretrained(from_model_path)


# Loading data
model_name = "t5-large"
tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
collator = data.OKvqaCollator(text_maxlength, tokenizer)

# Note: Edited function to support 'img.jpg' instead of downloading whole COCO dataset
eval_examples = data.load_okvqa_data(HOME_DIR / "val2014", split_type="val2014", use_gpt=True)

dataset = data.OkvqaDataset(eval_examples, n_context)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=per_gpu_batch_size,
    drop_last=False,
    num_workers=num_workers,
    collate_fn=collator,
)


# Tracing/saving model
answers = []
model = model.module if hasattr(model, "module") else model
model.eval()

with torch.no_grad():
    for batch in dataloader:
        (_, _, _, _, context_ids, context_mask) = batch

        example_input = (context_ids, context_mask)
        inputs = {"generate": example_input}
        scripted_model = torch.jit.trace_module(
            model, inputs
        )  # Note: RuntimeError: output with shape [] doesn't match the broadcast shape [1]
        torch.jit.save(scripted_model, to_model_path)


# Upload torchscript model to W&B
with wandb.init(job_type="stage", project=TO_PROJECT, dir=LOG_DIR):
    staged_at = wandb.Artifact(to_model_path, type=STAGED_MODEL_TYPE)
    staged_at.add_file(to_model_path)
    wandb.log_artifact(staged_at)
