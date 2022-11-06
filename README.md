# [admirer](https://admirer.loca.lt/)
<img width="1525" alt="Screen Shot 2022-10-13 at 9 30 21 PM" src="https://user-images.githubusercontent.com/40700820/195763037-1f5ca861-3eac-4338-8785-f6f16da79ad5.png">

# Description
- A full-stack ML-powered website that utilizes users’ webcam feeds to answer open-ended questions requiring outside knowledge.
- The website and repository together serve as an open-source demonstration and implementation of a visual question-answering model in a full-stack machine learning product. The visual question-answering pipeline is inspired by a paper from Microsoft; in short, we prompt GPT-3 with a generated image caption and object tag list, the question-answer pair, and context examples that demonstrate the task at hand in a few-shot learning method.
- The MVP of the website was built by Andrew Hinh as a top-25 final project for the FSDL 2022 course in which only the deployment code was written. The project was continued by The Transformees (Andrew Hinh and Aleks Hiidenhovi, a FSDL alum) as a submission for the ZenML Month of MLOps Competition. The data management, model development, testing, and continual learning scripts were additionally developed in the alloted time.

# Production
To setup the production server for the website in an AWS EC2 (free!) instance, we simply run:
1. `python3 app_gradio/app.py --flagging --model_url=https://joiajq6syp65ueonto4mswttzu0apfbi.lambda-url.us-west-1.on.aws/` to setup the Gradio app with an AWS Lambda backend.
2. `lt --port 11700 --subdomain admirer` to serve the Gradio app over a permanent localtunnel link.
3. `. ./backend_setup/deploy.sh` to implement continual learning by updating the AWS Lambda backend when signaled by a pushed commit to the repo and checking if the overall performance of the pipeline has improved.
    - Note: we measured performance using an "expressive-biased" accuracy, meaning we give GPT-3 the point if it returns the right answer regardless of a few filler words, and don't if it returns either the wrong answer or too undescriptive of an answer.

# Development
## Setup
1. Clone repo.
2. Follow the steps listed [here](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs/tree/main/setup#local), replacing the following commands:
    - ~~git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs.git~~ -> git clone https://github.com/andrewhinh/admirer.git
    - ~~cd fsdl-text-recognizer-2022-labs~~ -> cd admirer
    - ~~conda activate fsdl-text-recognizer-2022~~ -> conda activate admirer
3. If you're using a newer NVIDIA RTX GPU, run `pip3 install torch torchvision torchaudio` then `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`.
4. Sign up for OpenAI's API to get an API key, populate a `.env` file with your OpenAI API key in the format of `.env.template`, and reactivate (just activate again) the environment.
5. Sign up for a Weights and Biases account and run `python ./training/stage_model --fetch` to download the model weights and GPT-3 context examples locally.
## Notes
- The repo is separated into folders that each describe a part of the ML-project lifecycle:
    - `api_serverless`: the backend handler code using `AWS Lambda`.
    - `app_gradio`: the frontend code using `Gradio`.
    - `backend_setup`: the `AWS Lambda` backend setup and continuous deployment code.
    - `data_manage`: the data management code using `AWS S3` for data and `ZenML` log storage, `boto3` for data exploration, and `ZenML` + `Great Expectations` for data validation.
    - `load_test`: the load testing code using `Locust`.
    - `monitoring`: the model monitoring code using `Gradio's` flagging feature.
    - `question_answer`: the inference code.
    - `tasks`: the pipeline testing code.
    - `training` for the model development code using `PyTorch`, `PyTorch Lightning`, and `Weights and Biases`.
- From the main directory, run:
    - `python ./training/run_experiment --help` to learn more about how model training experiments can be conducted and configured depending on the use case.
    - `python app_gradio/app.py --flagging` to start a local Gradio app.
    - `python -c "from app_gradio.tests.test_app import test_local_run; test_local_run()"` to test the Gradio frontend by launching and pinging the frontend locally.
    - `. ./training/tests/test_memorize_caption.sh` to test the caption model's ability to memorize a single batch.
    - the shell scripts in `tasks` to test various aspects of the model pipeline (Ex: `. ./tasks/REPLACE`, replacing `REPLACE` with the corresponding shell script path).
- The other files and folders support the main folders in various ways from storing configurations to workflow scripts.

# Credit
- GI4E for their [database](https://www.unavarra.es/gi4e/databases/gi4e/?languageId=1) and [Scale AI](https://scale.com/) for their annotations.
- Facebook for their [image segmentation model](https://huggingface.co/facebook/detr-resnet-50-panoptic).
- NLP Connect for their [image caption model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning).
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and their [GPT-3 API](https://openai.com/api/).
- Microsoft for their [visual question answering code](https://github.com/microsoft/PICa).
