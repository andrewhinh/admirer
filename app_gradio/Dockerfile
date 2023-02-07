# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM python:3.7-buster

# Create the working directory
#   set -x prints commands and set -e causes us to stop on errors
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==21.1.3
RUN pip install -r requirements.txt
ENV PYTHONPATH ".:"

# Copy only the relevant directories
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY question_answer/ ./question_answer
COPY app_gradio/ ./app_gradio
COPY ./.aws/config ./.aws/config
COPY .env .env

# Use docker run -it --rm -p$PORT:11700 to run the web server and listen on host $PORT
#   add --help top see help for the Python script
ENTRYPOINT ["python3", "app_gradio/app.py", "--flagging", "--model_url=https://joiajq6syp65ueonto4mswttzu0apfbi.lambda-url.us-west-1.on.aws/"]
