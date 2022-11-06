from tqdm.auto import tqdm
from typing import List, Optional, Union

import torch
import torch.nn.functional as F


def top_k_top_p_filtering(
    next_token_logits: torch.FloatTensor,
    top_k: Optional[float] = None,
    top_p: Optional[float] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.FloatTensor:
    if top_k is None:
        top_k = next_token_logits.shape[-1]
    if top_p is None:
        top_p = 1.0

    p, largest_p_idx = F.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
    cumulative_p = p.cumsum(dim=-1)
    threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
    idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()
    cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
    censored_p = (cumulative_p <= cutoffs[:, None]) * p
    renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)

    final_p = torch.zeros_like(next_token_logits)
    row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
    final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)

    return final_p


def generate_sentence_from_image(
    model, encoder_outputs, tokenizer, max_text_length: int, device, top_k: int, top_p: int
) -> List[str]:
    generated_so_far = torch.LongTensor([[tokenizer.bos_token_id]] * len(encoder_outputs.last_hidden_state)).to(device)
    with torch.no_grad():
        for _ in tqdm(range(max_text_length)):
            attention_mask = torch.ones_like(generated_so_far)
            decoder_out = model(
                decoder_input_ids=generated_so_far,
                decoder_attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
            )

            next_token_logits = decoder_out["logits"][:, -1, :]
            filtered_p = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, device=device)
            next_token = torch.multinomial(filtered_p, num_samples=1)
            generated_so_far = torch.cat((generated_so_far, next_token), dim=1)

    return [tokenizer.decode(coded_sentence) for coded_sentence in generated_so_far]
