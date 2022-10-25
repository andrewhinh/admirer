# [admirer](https://admirer.loca.lt/)
<img width="1525" alt="Screen Shot 2022-10-13 at 9 30 21 PM" src="https://user-images.githubusercontent.com/40700820/195763037-1f5ca861-3eac-4338-8785-f6f16da79ad5.png">

A full-stack ML-powered website that utilizes users’ webcam feeds to answer open-ended questions requiring outside knowledge.

## Development Setup
1. Clone repo.
2. Follow the steps listed [here](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs/tree/main/setup).
3. Run `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` if you're using a newer NVIDIA RTX GPU.
4. Run `echo "REPLACE" >> ./question_answer/key.txt`, replacing `REPLACE` with your OpenAI API key and reactivate the environment.
5. Look at:
- `training/` for the model training, experiment tracking, and model staging scripts.
- `question_answer/` for the inference scripts.
- `app_gradio/` for the frontend scripts.
- `test.ipynb` for local testing of working with AWS S3 + LabelStudio, running Gradio, setting up AWS Lambda, and load testing with Locust.

## Notes
- Built as the final project for the FSDL 2022 course and a submission for the ZenML Month of MLOps Competition.
- Built using PyTorch, AWS Lambda, and ZenML among other tools.

## Credit
- GI4E for their [database](https://www.unavarra.es/gi4e/databases/gi4e/?languageId=1).
- Facebook for their [image segmentation model](https://huggingface.co/facebook/detr-resnet-50-panoptic).
- NLP Connect for their [image caption model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning).
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and their [GPT-3 API](https://openai.com/api/).
- Microsoft for their [visual question answering code](https://github.com/microsoft/PICa).
