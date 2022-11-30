#!/bin/bash
echo "Getting entity and sweep ID"


# Setting default values
DEFAULT_PROJECT="admirer-training"
DEFAULT_SWEEP_CONFIG="training/main-sweep.yaml"


# Getting arguments through flags
while getopts p:c: flag
do
    case "${flag}" in
        p) project=${OPTARG};;
        c) config=${OPTARG};;
        *);;
    esac
done

# Setting project and config values
if [ -z "${project}" ]; then
    PROJECT=$DEFAULT_PROJECT
else
    PROJECT=${project}
fi

if [ -z "${config}" ]; then
    SWEEP_CONFIG=$DEFAULT_SWEEP_CONFIG
else
    SWEEP_CONFIG=${config}
fi


# Getting entity and sweep ID
OUTPUT=$(python training/sweep_setup.py --project "$PROJECT" --config "$SWEEP_CONFIG")
OUTPUT=$(echo "$OUTPUT" | cut -d' ' -f3 | sed -n '2p')
OUTPUT=$(python training/process_setup_output.py --url "$OUTPUT")
ENTITY="$(echo "$OUTPUT" | cut -d' ' -f1)"
SWEEP_ID="$(echo "$OUTPUT" | cut -d' ' -f2)"
echo "$PROJECT"
echo "$ENTITY"
echo "$SWEEP_ID"


# Set the environment variables using:
# export PROJECT=${PROJECT}; export ENTITY=${ENTITY}; export SWEEP_ID=${SWEEP_ID}
# Start a tmux and for every GPU, change GPU_IDX accordingly, create a new window, and run:
# CUDA_VISIBLE_DEVICES=GPU_IDX wandb agent --project ${PROJECT} --entity ${ENTITY} ${SWEEP_ID}