# [admirer](https://admirer.loca.lt/)
<img width="1525" alt="Screen Shot 2022-10-13 at 9 30 21 PM" src="https://user-images.githubusercontent.com/40700820/195763037-1f5ca861-3eac-4338-8785-f6f16da79ad5.png">

# Description
- A full-stack ML-powered website that utilizes users’ webcam feeds to answer open-ended questions requiring outside knowledge.
- The website and repository together serve as an open-source demonstration and implementation of a visual question-answering model in a full-stack machine learning product. The visual question-answering pipeline is inspired by a paper from Microsoft; in short, we prompt GPT-3 with a generated image caption and object tag list, the question-answer pair, and context examples that demonstrate the task at hand in a few-shot learning method, achieving a [BERTScore](https://torchmetrics.readthedocs.io/en/stable/text/bert_score.html) computed F1 score of around .989 on the test set.
- The MVP of the website was built by Andrew Hinh as a top-25 final project for the FSDL 2022 course in which only the deployment code was written. The project was continued by The Transformees (Andrew Hinh and Aleks Hiidenhovi, a FSDL alum) as a submission for the ZenML Month of MLOps Competition. The data management, model development, testing, and continual learning scripts were additionally developed in the alloted time.

# Production
To setup the production server for the website in an AWS EC2 instance, we install basic packages such as `pip`, pull the repo, do some package setup, and simply run:
1. `python3 app_gradio/app.py --flagging --model_url=https://joiajq6syp65ueonto4mswttzu0apfbi.lambda-url.us-west-1.on.aws/` to setup the Gradio app with an AWS Lambda backend.
2. `lt --port 11700 --subdomain admirer` to serve the Gradio app over a permanent localtunnel link.
3. `. ./backend_setup/deploy.sh` to implement continual learning by updating the AWS Lambda backend when signaled by a pushed commit to the repo and checking if the BERTScore computed F1 score of the pipeline has improved.

# Development
## Setup
1. Clone repo.
2. Follow the steps listed [here](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs/tree/main/setup#local), replacing the following commands:
    - ~~git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs.git~~ -> git clone https://github.com/andrewhinh/admirer.git
    - ~~cd fsdl-text-recognizer-2022-labs~~ -> cd admirer
    - ~~conda activate fsdl-text-recognizer-2022~~ -> conda activate admirer
3. If you're using a newer NVIDIA RTX GPU, run `pip3 uninstall torch torchvision torchaudio` then `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`.
4. Sign up for OpenAI's API [here](https://openai.com/api/) to get an API key.
5. Sign up for an AWS account and get your account ID found at the beginning of the line below "Repository Name" [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2).
6. Populate a `.env` file with your OpenAI API key and AWS account ID in the format of `.env.template` and reactivate (just activate again) the environment.
7. Sign up for a Weights and Biases account [here](https://wandb.ai/signup), follow the steps after running `wandb login`, and run `python ./training/stage_model --fetch --from_project admirer` to download the models and context examples locally.
## Notes
- The repo is separated into folders that each describe a part of the ML-project lifecycle, some of which contain notebooks that allow for interaction with these components:
    - `api_serverless`: the backend handler code using `AWS Lambda`.
    - `app_gradio`: the frontend code using `Gradio`.
    - `backend_setup`: the `AWS Lambda` backend setup and continuous deployment code.
    - `data_manage`: the data management code using `AWS S3` for data and `ZenML` log storage, `boto3` for data exploration, and `ZenML` + `Great Expectations` for data validation.
    - `load_test`: the load testing code using `Locust`.
    - `monitoring`: the model monitoring code using `Gradio's` flagging feature.
    - `question_answer`: the inference code.
    - `tasks`: the pipeline testing code.
    - `training` for the model development code using `PyTorch`, `PyTorch Lightning`, and `Weights and Biases`.
- From the main directory, run (keeping in mind that there are more arguments that can be specified when running the python files than are listed):
    - `python ./training/run_experiment --help` to learn more about how model training experiments can be conducted and configured depending on the use case.
    - `python training/sweep.py` to start a W&B hyperparameter optimization sweep.
    - `python app_gradio/app.py --flagging` to start a local Gradio app.
    - `python -c "from app_gradio.tests.test_app import test_local_run; test_local_run()"` to test the Gradio frontend by launching and pinging the frontend locally.
    - `. ./training/tests/test_memorize_caption.sh` to test the caption model's ability to memorize a single batch.
    - the shell scripts in `tasks` to test various aspects of the model pipeline (Ex: `. ./tasks/REPLACE`, replacing `REPLACE` with the corresponding shell script path).
- The other files and folders support the main folders in various ways from storing configurations to workflow scripts.

# Credit
- GI4E for their [database](https://www.unavarra.es/gi4e/databases/gi4e/?languageId=1) and [Scale AI](https://scale.com/) for their annotations.
- Facebook for their [image segmentation model](https://huggingface.co/facebook/detr-resnet-50-panoptic).
- NLP Connect for their [base image caption model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) and Sachin Abeywardana for his [fine-tuning code](https://sachinruk.github.io/blog/pytorch/huggingface/2021/12/28/vit-to-gpt2-encoder-decoder-model.html).
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and their [GPT-3 API](https://openai.com/api/).
- Microsoft for their [visual question answering code](https://github.com/microsoft/PICa).
