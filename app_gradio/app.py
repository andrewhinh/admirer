"""Provide a webcam screenshot and a burning question and get back a lover-like accurate answer!"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import gradio as gr
from PIL import ImageStat
from PIL.Image import Image
import requests

from question_answer import util
from question_answer.answer import Pipeline


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FAVICON = APP_DIR / "logo.jpeg"  # path to a small image for display in browser tab and social media
README = APP_DIR / "README.md"  # path to an app readme file in HTML/markdown

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(predictor.run, flagging=args.flagging)
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def make_frontend(fn: Callable[[Image, str], str], flagging: bool = False):
    """Creates a gradio.Interface frontend for an image + text to text function."""
    img_examples_dir = Path("question_answer") / "tests" / "support" / "images"
    img_example_fnames = [elem for elem in os.listdir(img_examples_dir) if elem.endswith(".jpg")]
    img_example_paths = [img_examples_dir / fname for fname in img_example_fnames]
    img_example_paths = sorted(img_example_paths)

    question_examples_dir = Path("question_answer") / "tests" / "support" / "questions"
    question_example_fnames = [elem for elem in os.listdir(question_examples_dir) if elem.endswith(".txt")]
    question_example_paths = [question_examples_dir / fname for fname in question_example_fnames]
    question_example_paths = sorted(question_example_paths)

    questions = []
    for path in question_example_paths:
        with open(path, "r") as f:
            questions.append(f.readline())

    examples = [[str(img_path), question] for img_path, question in zip(img_example_paths, questions)]

    allow_flagging = "never"
    if flagging:  # logging user feedback to a local CSV file
        allow_flagging = "manual"
        flagging_callback = gr.CSVLogger()
        flagging_dir = "flagged"
    else:
        flagging_callback, flagging_dir = None, None

    readme = _load_readme(with_logging=allow_flagging == "manual")

    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=gr.components.Textbox(),  # what output widgets does it need? the default text widget
        # what input widgets does it need? we configure an image widget
        inputs=[
            gr.components.Image(type="pil", label="Webcam Image", source="webcam"),
            gr.components.Textbox(label="Question"),
        ],
        title="Admirer",  # what should we display at the top of the page?
        thumbnail=FAVICON,  # what should we display when the link is shared, e.g. on social media?
        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=examples,  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
        flagging_options=["incorrect", "offensive", "other"],  # what options do users have for feedback?
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = Pipeline()
            self._predict = model.predict

    def run(self, image, question):
        pred, metrics = self._predict_with_metrics(image, question)
        self._log_inference(pred, metrics)
        return pred

    def _predict_with_metrics(self, image, question):
        pred = self._predict(image, question)

        stats = ImageStat.Stat(image)
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        return pred, metrics

    def _predict_from_endpoint(self, image, question):
        """Send an image and question to an endpoint that accepts JSON and return the predicted text.

        The endpoint should expect a base64 representation of the image, encoded as a string,
        under the key "image" and a str representation of the question. It should return the predicted text under the key "pred".

        Parameters
        ----------
        image
            A PIL image of handwritten text to be converted into a string

        question
            A string containing the user's question

        Returns
        -------
        pred
            A string containing the predictor's guess of the text in the image.
        """
        encoded_image = util.encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps(
            {"image": "data:image/jpg;base64," + encoded_image, "question": "data:question/str;str," + question}
        )

        try:
            response = requests.post(self.url, data=payload, headers=headers, timeout=15)  # seconds, imposed by Heroku
            pred = response.json()["pred"]
        except requests.exceptions.Timeout:
            pred = "Sorry, the model took too long to respond. Please try again."

        return pred

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            lines = lines[: lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request as JSON with the key 'image'. Default is None, which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--flagging",
        action="store_true",
        help="Pass this flag to allow users to 'flag' model behavior and provide feedback.",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
