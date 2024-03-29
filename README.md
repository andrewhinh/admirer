# admirer

![demo](./assets/demo.png)

## Contents

- [admirer](#admirer)
  - [Contents](#contents)
  - [The Transformees](#the-transformees)
  - [Description](#description)
    - [Inference Pipeline](#inference-pipeline)
    - [Usage](#usage)
  - [Production](#production)
  - [Development](#development)
    - [Contributing](#contributing)
    - [Setup](#setup)
    - [Repository Structure](#repository-structure)
    - [Testing](#testing)
    - [Code Style](#code-style)
  - [Credit](#credit)

## The Transformees

1. [Andrew Hinh](https://github.com/andrewhinh)
2. [Aleks Hiidenhovi](https://github.com/alekshiidenhovi)

## Description

A website that uses webcam feeds to answer open-ended questions requiring outside knowledge. For more info, check out the ZenML [blog post](https://bit.ly/3BZ4YpB).

### Inference Pipeline

![inference_pipeline](./assets/inference_pipeline.png)

The visual question-answering pipeline is inspired by the paper from Microsoft linked in the [credit section](#credit). In short, we prompt GPT-3 with a generated image caption and object tag list, the question-answer pair, and context examples that demonstrate the task at hand in a few-shot learning method, achieving a [BERTScore](http://bit.ly/3tM1mmc) computed F1 score of around .989 on the test set.

### Usage

As a direct consequence of not feeding the image data directly to GPT-3, the best queries involve asking descriptive, counting, or similar questions about one or more objects visible in the background. For example, if there are two people in the image, one wearing a hat and the other wearing glasses, questions that would work well could include the following:

- "How many people are in the room?"
- "What color is the hat in the picture?"
- "How many people are wearing glasses?"

## Production

To setup the production server for the website, we:

1. Create an AWS Lambda function for the backend:

    ```bash
    . deploy/aws_login.sh
    python deploy/aws_lambda.py
    ```

2. Implement continual development by updating the AWS Lambda backend whenever a commit is pushed to the repo and the BERTScore computed F1 score of the pipeline has improved:

    ```bash
    . deploy/cont_deploy.sh
    ```

## Development

### Contributing

To contribute, check out the [guide](./CONTRIBUTING.md).

### Setup

1. Install conda if necessary:

    ```bash
    # Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
    # If on Windows, install chocolately: https://chocolatey.org/install. Then, run:
    # choco install make
    ```

2. Create the conda environment locally:

    ```bash
    cd admirer
    make conda-update
    conda activate admirer
    make pip-tools
    export PYTHONPATH=.
    echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
    ```

3. Install pre-commit:

    ```bash
    pre-commit install
    ```

4. Sign up for an OpenAI account and get an API key [here](https://beta.openai.com/account/api-keys).
5. Populate a `.env` file with your key and the backend URL in the format of `.env.template`, and reactivate the environment.
6. Sign up for a Weights and Biases account [here](https://wandb.ai/signup) and download the CLIP ONNX file locally:

    ```bash
    wandb login
    python ./training/stage_model.py --fetch --from_project admirer
    ```

7. (Optional) Sign up for an AWS account [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2) and set up your AWS credentials locally, referring to [this](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config) as needed:

    ```bash
    aws configure
    ```

If the instructions aren't working for you, head to [this Google Colab](https://colab.research.google.com/drive/1Z34DLHJm1i1e1tnknICujfZC6IaToU3k?usp=sharing), make a copy of it, and run the cells there to get an environment set up.

### Repository Structure

The repo is separated into main folders that each describe a part of the ML-project lifecycle, some of which contain interactive notebooks, and supporting files and folders that store configurations and workflow scripts:

```bash
.
├── api_serverless  # the backend handler code using AWS Lambda.
├── app_gradio      # the frontend code using Gradio.
├── deploy   # the AWS Lambda backend setup and continuous deployment code.
├── data_manage     # the data management code using AWS S3 for training data and ZenML log storage, boto3 for data exploration, and ZenML + Great Expectations for data validation.
├── load_test       # the load testing code using Locust.
├── monitoring      # the model monitoring code using Gradio's flagging feature.
├── question_answer # the inference code.
├── tasks           # the pipeline testing code.
├── training        # the model development code using PyTorch, PyTorch Lightning, and Weights and Biases.
```

### Testing

From the main directory, there are various ways to test the pipeline:

- To start a W&B hyperparameter optimization sweep for the caption model (on one GPU):

    ```bash
    . ./training/sweep/sweep.sh
    CUDA_VISIBLE_DEVICES=0 wandb agent --project ${PROJECT} --entity ${ENTITY} ${SWEEP_ID}
    ```

- To train the caption model (add `--strategy ddp_find_unused_parameters_false` for multi-GPU machines; takes ~7.5 hrs on an 8xA100 Lambda Labs instance):

    ```bash
    python ./training/run_experiment.py \
    --data_class PICa --model_class ViT2GPT2 --gpus "-1" \
    --wandb --log_every_n_steps 25 --max_epochs 300 \
    --augment_data True --num_workers "$(nproc)" \
    --batch_size 2 --one_cycle_max_lr 0.01 --top_k 780 --top_p 0.65 --max_label_length 50
    ```

- To test the caption model (best model can be downloaded from [here](https://wandb.ai/admirer/admirer-training/artifacts/model/model-2vgqajre/v4/files)):

    ```bash
    python ./training/test_model.py \
    --data_class PICa --model_class ViT2GPT2 \
    --num_workers "$(nproc)" --load_checkpoint training/model.pth
    ```

- To start the app locally (uncomment code in PredictorBackend.__init__ and set use_url=False to use the local model instead of the API):

    ```bash
    python app_gradio/app.py
    ```

- To test the Gradio frontend by launching and pinging the frontend locally:

    ```bash
    python -c "from app_gradio.tests.test_app import test_local_run; test_local_run()"
    ```

- To test the caption model's ability to memorize a single batch:

    ```bash
    . ./training/tests/test_memorize_caption.sh
    ```

- To run integration tests for the model pipeline:

    ```bash
    . ./tasks/integration_test.sh
    ```

- To run unit tests for the model pipeline:

    ```bash
    . ./tasks/unit_test.sh
    ```

- To test the whole model pipeline:

    ```bash
    . ./tasks/test.sh
    ```

### Code Style

- To lint your code:

    ```bash
    pre-commit run --all-files
    ```

## Credit

- GI4E for their [database](https://www.unavarra.es/gi4e/databases/gi4e/?languageId=1) and [Scale AI](https://scale.com/) for their annotations.
- Facebook for their [image segmentation model](https://huggingface.co/facebook/detr-resnet-50-panoptic).
- NLP Connect for their [base image caption model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) and Sachin Abeywardana for his [fine-tuning code](https://sachinruk.github.io/blog/pytorch/huggingface/2021/12/28/vit-to-gpt2-encoder-decoder-model.html).
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and [GPT-3 API](https://openai.com/api/).
- Microsoft for their [visual question answering code](https://github.com/microsoft/PICa).
