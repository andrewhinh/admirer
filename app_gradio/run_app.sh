#!/bin/bash
AWS_REGION=$(aws configure get region)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account | sed 's/"//g')
while ! docker run -it --rm "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/admirer-frontend:latest"; do
    sleep 1
done