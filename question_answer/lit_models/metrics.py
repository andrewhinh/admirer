"""Special-purpose metrics for tracking our model performance."""
from pathlib import Path

import torchmetrics

BERT_SCORE_PATH = (
    Path(__file__).resolve().parents[2] / "question_answer" / "artifacts" / "answer" / "transformers" / "bert_score"
)


class BertF1Score(torchmetrics.text.bert.BERTScore):
    """Character error rate metric, allowing for tokens to be ignored."""

    def __init__(self, model_type=BERT_SCORE_PATH):
        super().__init__(model_type)

    def __call__(self, preds, targets):
        f1s = super().__call__(preds, targets)["f1"]
        return sum(f1s) / len(f1s)


def test_bert_f1_score():
    bert_f1 = BertF1Score()
    preds = ["hello there", "general kenobi"]
    target = ["hello there", "master kenobi"]
    f1 = bert_f1(preds, target)
    ex_f1s = [0.9999998807907104, 0.9960542917251587]  # On main page of torchmetrics page for BERTScore
    assert f1 == sum(ex_f1s) / len(ex_f1s)


if __name__ == "__main__":
    test_bert_f1_score()
