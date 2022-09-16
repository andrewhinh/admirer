# Libraries
from pathlib import Path

import model as md
import wandb


# Useful variables
PRETRAINED_MODEL_PATH = "Desktop/Projects/large_both_knowledge/"
STAGED_MODEL_FILENAME = "model.pt"
TO_PROJECT = "admirer-training"
LOG_DIR = Path("training") / "logs"
STAGED_MODEL_TYPE = "prod-ready"

from_model_path = Path("/home/andrewhinh/") / PRETRAINED_MODEL_PATH
to_model_path = Path("./") / STAGED_MODEL_FILENAME


# Load best pre-trained model from https://github.com/guilk/KAT
model_class = md.FiDT5
model = model_class.from_pretrained(from_model_path)

"""
Trying to find example input for model.generate()
# Tracing/saving model
# scripted_model = torch.jit.script(model) #Scripting doesn't work b/c of generator functions (ex. any(), etc.)
example_forward_input = torch.rand(1, 1, 3, 3) #Find shape of example_input for md.FiDT5.generate()
scripted_model = torch.jit.trace(model.generate, example_forward_input)
torch.jit.save(scripted_model, to_model_path)
"""

# Upload torchscript model to W&B
with wandb.init(job_type="stage", project=TO_PROJECT, dir=LOG_DIR):
    staged_at = wandb.Artifact(to_model_path, type=STAGED_MODEL_TYPE)
    staged_at.add_file(to_model_path)
    wandb.log_artifact(staged_at)
