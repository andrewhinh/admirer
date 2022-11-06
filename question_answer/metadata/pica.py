from pathlib import Path
import question_answer.metadata.shared as shared


ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "question_answer" / "artifacts" / "answer"
RAW_DATA_DIRNAME = ARTIFACT_PATH / "coco_annotations"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "pica"

NUM_ORIGINAL_EXAMPLES = 9009
NUM_ADDED_EXAMPLES = 1236
NUM_TEST_EXAMPLES = 36

TRAIN_VAL_SPLIT = 0.9
NUM_TRAINVAL = NUM_ADDED_EXAMPLES - NUM_TEST_EXAMPLES
NUM_TRAIN_EXAMPLES = NUM_TRAINVAL * TRAIN_VAL_SPLIT
NUM_VAL_EXAMPLES = NUM_TRAINVAL * (1 - TRAIN_VAL_SPLIT)

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224  # Originally = 600, 800
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

MAX_LABEL_LENGTH = 50

DIMS = (3, IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIMS = (MAX_LABEL_LENGTH, 1)