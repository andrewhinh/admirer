#!/bin/bash
while :
do
    git fetch
    if [ "$(git rev-list HEAD...origin/main --count)" != 0 ]; then
        echo "Checking if backend needs to be updated..."
        git pull

        python3 ./training/stage_model.py --fetch

        BEST_ACC=$(< ./question_answer/evaluation/best_pica_acc.txt)
        NEW_ACC=$(python3 ./question_answer/evaluation/evaluate_pica.py)
        if [ "$NEW_ACC" \> "$BEST_ACC" ]; then
            echo "Updating backend..."
            echo "$NEW_ACC" >| ./question_answer/evaluation/best_pica_acc.txt
            python3 ./backend_setup/deploy.py
        else
            echo "No improvement -> no updates made"

        rm -rf ./question_answer/artifacts
        fi
    fi
done
exit 0