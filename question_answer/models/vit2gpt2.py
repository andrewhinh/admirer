import argparse
from pathlib import Path

from transformers import EncoderDecoderModel, GPT2Tokenizer
import torch.nn as nn

SAVE_PATH = Path(__file__).resolve().parents[2] / "question_answer" / "artifacts" / "answer" / "transformers"
VIT_MODEL = "google/vit-base-patch16-224-in21k"
DISTIL_GPT2 = "distilgpt2"


class ViT2GPT2(nn.Module):
    """Pass an image through a ViT and decode the resulting embedding with GPT-2."""

    def __init__(
        self,
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        # Arguments
        self.args = vars(args) if args is not None else {}
        self.encoder_path = self.args.get("encoder_path", SAVE_PATH / VIT_MODEL)
        self.decoder_and_tokenizer_path = self.args.get("decoder_and_tokenizer_path", SAVE_PATH / DISTIL_GPT2)

        # model
        self.vit2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained(
            self.encoder_path, self.decoder_and_tokenizer_path
        )

        # tokenizer
        # make sure GPT2 appends EOS in begin and end
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return outputs

        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.decoder_and_tokenizer_path)
        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token
        self.gpt2_tokenizer = gpt2_tokenizer

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--encoder_path", type=Path, default=SAVE_PATH / VIT_MODEL)
        parser.add_argument("--decoder_and_tokenizer_path", type=Path, default=SAVE_PATH / DISTIL_GPT2)
        return parser
