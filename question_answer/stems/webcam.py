"""PICa Stem class."""
import torchvision.transforms as transforms

import question_answer.metadata.pica as metadata
from question_answer.stems.image import ImageStem


IMAGE_HEIGHT, IMAGE_WIDTH = metadata.IMAGE_HEIGHT, metadata.IMAGE_WIDTH
IMAGE_SHAPE = metadata.IMAGE_SHAPE
MEAN, STD = 0.5, 0.5


class WebcamStem(ImageStem):
    """A stem for handling webcam screenshots."""

    def __init__(
        self,
        augment=False,
    ):
        super().__init__()

        if not augment:
            self.pil_transforms = transforms.Compose([transforms.Resize(IMAGE_SHAPE)])
        else:
            # IMAGE_SHAPE is (600, 800)
            self.pil_transforms = transforms.Compose(
                [
                    transforms.Resize(IMAGE_SHAPE),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.RandomRotation(degrees=20)], p=0.1),
                ]
            )
        self.torch_transforms = transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
