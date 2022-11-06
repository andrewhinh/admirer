#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

CI="${CI:-false}"
if [ "$CI" = false ]; then
  export WANDB_PROJECT="admirer"
else
  export WANDB_PROJECT="admirer-ci"
fi

echo "training smaller version of real model class on real data"
python training/run_experiment.py --data_class=PICa --model_class=ViT2GPT2 \
  --batch_size 2 --lr 0.0001 \
  --limit_train_batches 1 --limit_val_batches 1 --limit_test_batches 1 --num_sanity_val_steps 0 \
  --num_workers 1 --wandb || FAILURE=true

TRAIN_RUN=$(find ./training/logs/wandb/latest-run/* | grep -Eo "run-([[:alnum:]])+\.wandb" | sed -e "s/^run-//" -e "s/\.wandb//")

echo "staging trained model from run $TRAIN_RUN"
python training/stage_model.py --run "$TRAIN_RUN" --staged_model_name test-dummy --ckpt_alias latest || FAILURE=true

echo "fetching staged model"
python training/stage_model.py --fetch --staged_model_name test-dummy || FAILURE=true
STAGE_RUN=$(find ./training/logs/wandb/latest-run/* | grep -Eo "run-([[:alnum:]])+\.wandb" | sed -e "s/^run-//" -e "s/\.wandb//")

if [ "$FAILURE" = true ]; then
  echo "Model development test failed"
  echo "cleaning up local files"
  rm -rf question_answer/artifacts/test-dummy
  echo "leaving remote files in place"
  exit 1
fi
echo "cleaning up local and remote files"
rm -rf question_answer/artifacts/test-dummy
python training/cleanup_artifacts.py --run_ids "$TRAIN_RUN" "$STAGE_RUN" --all -v
# note: if $TRAIN_RUN and $STAGE_RUN are not set, this will fail.
#  that's good because it avoids all artifacts from the project being deleted due to the --all.
echo "Model development test passed"
exit 0
