# Admirer: Open-Ended VQA Requiring Outside Knowledge
### November 17th, 2022 - Andrew Hinh and Aleks Hiidenhovi

<img width="1525" alt="Screen Shot 2022-10-13 at 9 30 21 PM" src="https://user-images.githubusercontent.com/40700820/195763037-1f5ca861-3eac-4338-8785-f6f16da79ad5.png">

For the ZenML Month of MLOps Competition, we created [Admirer](https://admirer.loca.lt/), a full-stack ML-powered website that utilizes users’ webcam feeds to answer open-ended questions requiring outside knowledge. Andrew built the MVP of the website as a [top-25 final project](https://bit.ly/3h8CqlX) for the [FSDL 2022 course](https://bit.ly/3NYNf6v) in which only the deployment code was written. The project was continued by [The Transformees](#the-transformees) and won the `Most Promising Entry` prize for the [ZenML Month of MLOps Competition](https://bit.ly/3EmoCxv) in the [closing ceremony](https://bit.ly/3tsDi7V). The data management, model development, testing, and continual learning scripts were additionally developed in the allotted time.

## How to use it?
As a direct consequence of not feeding the image data directly to GPT-3, the best queries involve asking descriptive, counting, or similar questions about one or more objects visible in the background. For example, if there are two people in the image, one wearing a hat and the other wearing glasses, questions that would work well could include the following:
- "How many people are in the room?"
- "What color is the hat in the picture?"
- "How many people are wearing glasses?"

## How does it work?
![Screenshot_2022-11-18_at_11 17 22](https://user-images.githubusercontent.com/40700820/202741457-ef306fd8-27c6-47ed-89bb-913bb44bd312.png)

The visual question-answering pipeline is inspired by [this paper](https://github.com/microsoft/PICa) from Microsoft; in short, we prompt GPT-3 with a generated image caption and object tag list, the question-answer pair, and context examples that demonstrate the task at hand in a few-shot learning method as shown in the diagram above. As a result, we achieve a [BERTScore](https://torchmetrics.readthedocs.io/en/stable/text/bert_score.html) computed F1 score of around .989 on a test set we selected at random.

## Repo Structure
The [repo](https://github.com/andrewhinh/admirer) is separated into main folders that each describe a part of the ML-project lifecycle, some of which contain notebooks that allow for interaction with these components, and supporting files and folders that store configurations and workflow scripts:
```bash
.
├── api_serverless  # the backend handler code using AWS Lambda.
├── app_gradio      # the frontend code using Gradio.
├── backend_setup   # the AWS Lambda backend setup and continuous deployment code.
├── data_manage     # the data management code using AWS S3 for training data and ZenML log storage, boto3 for data exploration, and ZenML + Great Expectations for data validation.
├── load_test       # the load testing code using Locust.
├── monitoring      # the model monitoring code using Gradio's flagging feature.
├── question_answer # the inference code.
├── tasks           # the pipeline testing code.
├── training        # the model development code using PyTorch, PyTorch Lightning, and Weights and Biases.
```

## Production
From the repo structure, it’s easy to set up the production server for the website in an AWS EC2 instance. We simply:
1. Setup the instance: install packages such as `pip`, pull the repo, and install the environment requirements:
2. Setup the Gradio app with an AWS Lambda backend:
```bash
python3 app_gradio/app.py --flagging --model_url=https://joiajq6syp65ueonto4mswttzu0apfbi.lambda-url.us-west-1.on.aws/
```
3. Serve the Gradio app over a permanent localtunnel link:
```bash
. ./backend_setup/localtunnel.sh
```
4. Implement continual development by updating the AWS Lambda backend when signaled by a pushed commit to the repo and checking if the BERTScore computed F1 score of the pipeline has improved:
```bash
. ./backend_setup/cont_deploy.sh
```

## Let's connect!
If you’re interested in reading more about our development setup and process, check out the repo’s README [here](https://github.com/andrewhinh/admirer#readme). To reach out to us, you can head over to Andrew's [LinkedIn](https://www.linkedin.com/in/andrew-hinh/) or [GitHub](https://github.com/andrewhinh) and Alek's [LinkedIn](https://www.linkedin.com/in/hiidenhovi/?originalSubdomain=fi) or [GitHub](https://github.com/alekshiidenhovi).

## Show some love!
If you found our work interesting, please head over to [the repo](https://github.com/andrewhinh/admirer) and give us a star! Thanks for reading!
