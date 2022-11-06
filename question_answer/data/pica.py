"""PICa Dataset class."""
import argparse
import json
from pathlib import Path
import requests
from io import BytesIO
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch

from question_answer import util
from question_answer.data.base_data_module import BaseDataModule, load_and_print_info
from question_answer.data.util import BaseDataset
import question_answer.metadata.pica as metadata
from question_answer.stems.webcam import WebcamStem


IMAGE_SHAPE = metadata.IMAGE_SHAPE
RAW_DATA_DIRNAME = metadata.RAW_DATA_DIRNAME
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME

NUM_ADDED_EXAMPLES = metadata.NUM_ADDED_EXAMPLES
NUM_TRAINVAL = metadata.NUM_TRAINVAL
NUM_VAL_EXAMPLES = metadata.NUM_VAL_EXAMPLES

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


class PICa(BaseDataModule):
    """PICa webcam screenshots + annotations dataset."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true").lower() == "true"

        self.input_dims = metadata.DIMS  # We assert that this is correct in setup()
        self.output_dims = metadata.OUTPUT_DIMS  # We assert that this is correct in setup()

        self.transform = WebcamStem()
        self.trainval_transform = WebcamStem(augment=self.augment)

        self.data_file = RAW_DATA_DIRNAME / "admirer-pica.json"

        self.test_ids = self.calc_test_ids()
        self.validation_ids = self.calc_validation_ids()
        self.train_ids = self.calc_train_ids()
        self.all_ids = self.train_ids + self.validation_ids + self.test_ids

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        if (PROCESSED_DATA_DIRNAME / "_properties.json").exists():
            return
        rank_zero_info("PICa.prepare_data: Logging dataset info to a json file...")

        properties = {}
        for split in ["train", "val", "test"]:
            screenshots, captions = get_screenshots_and_captions(self, split=split)
            save_screenshots_and_captions(screenshots=screenshots, captions=captions, split=split)

            properties.update(
                {
                    id_: {
                        "image_shape": screenshots[id_].size[::-1],
                        "num_words": _num_words(caption),
                    }
                    for id_, caption in captions.items()
                }
            )

        with open(PROCESSED_DATA_DIRNAME / "_properties.json", "w") as f:
            json.dump(properties, f, indent=4)

    def load_image(self, id: str) -> Image.Image:
        """Load and return an image of a webcam screenshot."""
        url = self.screenshot_url_by_id(id)
        response = requests.get(url)
        return util.read_image_pil_file(BytesIO(response.content))

    def setup(self, stage: str = None) -> None:
        def _load_dataset(split: str, transform: Callable) -> BaseDataset:
            screenshots, captions = load_processed_crops_and_labels(split)
            return BaseDataset(screenshots, captions, transform=transform)

        rank_zero_info(f"PICa.setup({stage}): Loading PICa webcam screenshots and captions...")
        validate_input_and_output_dimensions(input_dims=self.input_dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(split="train", transform=self.trainval_transform)
            self.data_val = _load_dataset(split="val", transform=self.transform)

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", transform=self.transform)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = "PICa Dataset\n" f"Input dims : {self.input_dims}\n" f"Output dims: {self.output_dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y)}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt)}\n"
        )
        return basic + data

    def ids_by_split(self, split):
        return {"train": self.train_ids, "val": self.validation_ids, "test": self.test_ids}[split]

    def calc_train_ids(self):
        """A list of screenshot IDs which are in the training set."""
        return list(set(range(0, NUM_ADDED_EXAMPLES)) - (set(self.test_ids) | set(self.validation_ids)))

    def calc_validation_ids(self):
        """A list of screenshot IDs which are in the validation set."""
        ids = []
        while len(ids) < NUM_VAL_EXAMPLES:
            id = np.random.randint(low=0, high=int(NUM_TRAINVAL))
            if id in ids:
                continue
            else:
                ids.append(id)
        return ids

    def calc_test_ids(self):
        """A list of screenshot IDs which are in the test set."""
        return list(range(NUM_TRAINVAL, NUM_ADDED_EXAMPLES))

    def screenshot_url_by_id(self, id):
        """A dict mapping a screenshot id to its filename."""
        df = pd.read_json(self.data_file)
        return df.loc[id, "webcam"]

    def caption_by_id(self, id):
        """A dict mapping a screenshot id to its caption."""
        df = pd.read_json(self.data_file)
        return df.loc[id, "caption"]


def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["image_shape"]["max"]
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]

    # Add 2 because of start and end tokens
    assert output_dims is not None and output_dims[0] >= properties["num_words"]["max"] + 2


def get_screenshots_and_captions(dataset: PICa, split: str) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """Create screenshots + captions for a given split, with resizing."""
    screenshots = {}
    captions = {}
    ids = dataset.ids_by_split(split)
    for id in ids:
        image = dataset.load_image(id)
        screenshots[id] = image.resize(IMAGE_SHAPE)
        captions[id] = dataset.caption_by_id(id)
    assert len(screenshots) == len(captions)
    return screenshots, captions


def save_screenshots_and_captions(screenshots: Dict[str, Image.Image], captions: Dict[str, str], split: str):
    """Save crops, labels and shapes of crops of a split."""
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

    with open(_captions_filename(split), "w") as f:
        json.dump(captions, f, indent=4)

    for id_, crop in screenshots.items():
        crop.save(_screenshot_filename(id_, split))


def load_processed_crops_and_labels(split: str) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and labels for given split."""
    with open(_captions_filename(split), "r") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_screenshots = []
    ordered_captions = []
    for id_ in sorted_ids:
        image = Image.open(_screenshot_filename(id_, split))
        ordered_screenshots.append(image.convert(mode=image.mode))
        ordered_captions.append(labels[id_])

    assert len(ordered_screenshots) == len(ordered_captions)
    return ordered_screenshots, ordered_captions


def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]

    image_shapes = np.array(_get_property_values("image_shape"))
    aspect_ratios = image_shapes[:, 1] / image_shapes[:, 0]
    return {
        "num_words": {"min": min(_get_property_values("num_words")), "max": max(_get_property_values("num_words"))},
        "image_shape": {"min": image_shapes.min(axis=0), "max": image_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }


def _captions_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_captions.json"


def _screenshot_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"


def _num_words(caption: str) -> int:
    """Return number of words in caption."""
    word_list = caption.split()
    return len(word_list)


if __name__ == "__main__":
    load_and_print_info(PICa)
