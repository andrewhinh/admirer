"""Test submodules of the data module."""
import os
import shutil

import numpy as np
import pytest

from question_answer.data import pica
from question_answer.metadata.pica import TRAIN_VAL_SPLIT


@pytest.mark.data
class TestDataset:
    """Tests downloading and setup of a dataset."""


pica_dir = pica.PROCESSED_DATA_DIRNAME


@pytest.fixture(scope="module")
def pica_dataset():
    _remove_if_exist(pica_dir)
    dataset = pica.PICa()
    dataset.prepare_data()
    return dataset


def _exist(dir):
    return all(os.path.exists(dir))


def _remove_if_exist(dir):
    shutil.rmtree(dir, ignore_errors=True)


class TestPICa(TestDataset):
    """Tests downloading and properties of the dataset."""

    dir = pica_dir

    def test_prepare_data(self, pica_dataset):
        """Tests whether the prepare_data method has produced the expected directories."""
        assert _exist(self.dir)

    def test_setup(self, pica_dataset):
        """Tests features of the fully set up dataset."""
        dataset = pica_dataset
        dataset.setup()
        assert all(map(lambda s: hasattr(dataset, s), ["x_trainval", "y_trainval", "x_test", "y_test"]))
        splits = [dataset.x_trainval, dataset.y_trainval, dataset.x_test, dataset.y_test]
        assert all(map(lambda attr: type(attr) == np.ndarray, splits))
        observed_train_frac = len(dataset.data_train) / (len(dataset.data_train) + len(dataset.data_val))
        assert np.isclose(observed_train_frac, TRAIN_VAL_SPLIT)
        assert dataset.input_dims[-2:] == dataset.x_trainval[0].shape  # ToTensor() adds a dimension
        assert len(dataset.output_dims) == len(dataset.y_trainval.shape)  # == 1

    def test_iam_parsed_lines(self, pica_dataset):
        """Tests that we retrieve the same number of captions and screenshots."""
        for id in pica_dataset.all_ids:
            assert len(pica_dataset.caption_by_id[id]) == len(pica_dataset.screenshot_url_by_id[id])

    def test_iam_data_splits(self, pica_dataset):
        """Fails when any identifiers are shared between training, test, or validation."""
        assert not set(pica_dataset.train_ids) & set(pica_dataset.validation_ids)
        assert not set(pica_dataset.train_ids) & set(pica_dataset.test_ids)
        assert not set(pica_dataset.validation_ids) & set(pica_dataset.test_ids)
