# admirer
A full-stack ML-powered website that utilizes users’ webcam feeds to answer open-ended questions requiring outside knowledge.

## Development Setup
1. Clone repo.
2. Follow the steps listed [here](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs/tree/main/setup).
3. Run `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` if you're using newer NVIDIA RTX GPU.
4. Run `echo "REPLACE" >> ./question_answer/key.txt`, replacing `REPLACE` with your OpenAI API key and reactivate the environment.
5. Setting up AWS Lambda is in `test.ipynb`.

## Notes
- Built as the final project for the FSDL 2022 course and a submission for the ZenML Month of MLOps Competition.
- Built using PostgreSQL, PyTorch, AWS Lambda, Locust, and ZenML among other tools.

## Credit
- [Facebook](https://huggingface.co/facebook/detr-resnet-50-panoptic) for their image segmentation model.
- [NLP Connect](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) for their image caption model.
- OpenAI for their [text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and their [GPT-3 API](https://openai.com/api/).
- [Microsoft](https://github.com/microsoft/PICa) for their visual question answering code.
