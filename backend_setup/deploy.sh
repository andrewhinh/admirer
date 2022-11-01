#!/bin/bash

UPSTREAM=${1:-'@{u}'}
LOCAL=$(git rev-parse @)
BASE=$(git merge-base @ "$UPSTREAM")

while :
do
    git remote update
    if [ "$LOCAL" = "$BASE" ]; then
        echo "Updating backend"
        jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=None --execute ./backend_setup/continuous_deployment.ipynb
    fi
done

exit 0