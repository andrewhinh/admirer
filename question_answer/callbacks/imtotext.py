import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

from .util import check_and_warn
from question_answer.lit_models.util import generate_sentence_from_image
from torchvision import transforms


descale = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / 0.5),
        transforms.Normalize(mean=-0.5, std=[1.0, 1.0, 1.0]),
    ]
)


class ImageToTextTableLogger(pl.Callback):
    """Logs the inputs and outputs of an image-to-text model to Weights & Biases."""

    def __init__(self, max_images_to_log=32, on_train=True):
        super().__init__()
        self.max_images_to_log = min(max(max_images_to_log, 1), 32)
        self.on_train = on_train

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, output, batch, batch_idx):
        if self.on_train:
            if check_and_warn(trainer.logger, "log_table", "image-to-text table"):
                return
            else:
                self._log_image_text_table(trainer, output, batch, "train/predictions")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, module, output, batch, batch_idx, dataloader_idx):
        if check_and_warn(trainer.logger, "log_table", "image-to-text table"):
            return
        else:
            self._log_image_text_table(trainer, output, batch, "validation/predictions")

    def _log_image_text_table(self, trainer, output, batch, key):
        images, actual_sentences = batch
        trainer = trainer.model
        encoder_outputs = trainer.model.encoder(pixel_values=images.to(trainer.device))
        generated_sentences = generate_sentence_from_image(
            trainer.model,
            encoder_outputs,
            trainer.tokenizer,
            trainer.max_label_length,
            trainer.device,
            trainer.top_k,
            trainer.top_p,
        )
        images = [wandb.Image(transforms.ToPILImage()(descale(image))) for image in images]
        data = list(map(list, zip(images, actual_sentences, generated_sentences)))
        columns = ["Images", "Actual Sentence", "Generated Sentence"]
        table = wandb.Table(data=data, columns=columns)
        trainer.logger.experiment.log({f"epoch {trainer.current_epoch} results": table})


class ImageToTextCaptionLogger(pl.Callback):
    """Logs the inputs and outputs of an image-to-text model to Weights & Biases."""

    def __init__(self, max_images_to_log=32, on_train=True):
        super().__init__()
        self.max_images_to_log = min(max(max_images_to_log, 1), 32)
        self.on_train = on_train
        self._required_keys = ["gt_strs", "pred_strs"]

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, output, batch, batch_idx):
        if self.has_metrics(output):
            if check_and_warn(trainer.logger, "log_image", "image-to-text"):
                return
            else:
                self._log_image_text_caption(trainer, output, batch, "train/predictions")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, module, output, batch, batch_idx, dataloader_idx):
        if self.has_metrics(output):
            if check_and_warn(trainer.logger, "log_image", "image-to-text"):
                return
            else:
                self._log_image_text_caption(trainer, output, batch, "validation/predictions")

    @rank_zero_only
    def on_test_batch_end(self, trainer, module, output, batch, batch_idx, dataloader_idx):
        if self.has_metrics(output):
            if check_and_warn(trainer.logger, "log_image", "image-to-text"):
                return
            else:
                self._log_image_text_caption(trainer, output, batch, "test/predictions")

    def _log_image_text_caption(self, trainer, output, batch, key):
        xs, _ = batch
        gt_strs = output["gt_strs"]
        pred_strs = output["pred_strs"]

        mx = self.max_images_to_log
        xs, gt_strs, pred_strs = list(xs[:mx]), gt_strs[:mx], pred_strs[:mx]

        trainer.logger.log_image(key, xs, caption=pred_strs)

    def has_metrics(self, output):
        return all(key in output.keys() for key in self._required_keys)
