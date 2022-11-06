"""Test for answer module."""
import json
import os
from pathlib import Path
import time

from question_answer.answer import Pipeline
from question_answer.lit_models.metrics import BertF1Score


os.environ["CUDA_VISIBLE_DEVICES"] = ""


_FILE_DIRNAME = Path(__file__).parents[0].resolve()
_SUPPORT_DIRNAME = _FILE_DIRNAME / "support"
_IMAGES_SUPPORT_DIRNAME = _SUPPORT_DIRNAME / "images"
_QUESTIONS_SUPPORT_DIRNAME = _SUPPORT_DIRNAME / "questions"

# restricting number of samples to prevent CirleCI running out of time
_NUM_MAX_SAMPLES = 2 if os.environ.get("CIRCLECI", False) else 100


def test_answer():
    """Test Pipeline."""
    support_images = list(_IMAGES_SUPPORT_DIRNAME.glob("*.png"))
    support_questions = list(_QUESTIONS_SUPPORT_DIRNAME.glob("*.txt"))
    with open(_SUPPORT_DIRNAME / "data_by_file_id.json", "r") as f:
        support_data_by_file_id = json.load(f)

    start_time = time.time()
    pipeline = Pipeline()
    end_time = time.time()
    print(f"Time taken to initialize Pipeline: {round(end_time - start_time, 2)}s")

    for i, (support_image, support_question) in enumerate(zip(support_images, support_questions)):
        if i >= _NUM_MAX_SAMPLES:
            break
        expected_text = support_data_by_file_id[support_image.stem]["predicted_text"]
        start_time = time.time()
        predicted_text = _test_answer(support_image, support_question, expected_text, pipeline)
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)

        ground_truth = support_data_by_file_id[support_image.stem]["ground_truth_text"]
        f1 = BertF1Score()(predicted_text, ground_truth).item()
        print(
            f"Bert F1 score is {round(f1, 3)} for files {support_image.name} and {support_question.name} (time taken: {time_taken}s)"
        )


def _test_answer(image_filename: Path, expected_text: str, pipeline: Pipeline):
    """Test ParagraphTextRecognizer on 1 image."""
    predicted_text = pipeline.predict(image_filename)
    assert predicted_text == expected_text, f"predicted text does not match expected for {image_filename.name}"
    return predicted_text
