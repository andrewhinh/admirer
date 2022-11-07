#!/bin/bash
while :
do
    git fetch
    if [ "$(git rev-list HEAD...origin/main --count)" != 0 ]; then
        echo "Checking if backend needs to be updated..."
        git pull

        python3 ./training/stage_model.py --fetch --from_project admirer

        CURRENT_F1_PATH=./question_answer/evaluation/best_pica_f1.txt
        CURRENT_F1_SCORE=$(< "$CURRENT_F1_PATH")
        NEW_F1=$(python3 ./question_answer/evaluation/evaluate_pica.py)
        if [ "$NEW_F1" \> "$CURRENT_F1_SCORE" ]; then
            echo "Updating backend..."
            echo "$NEW_F1" >| "$CURRENT_F1_PATH"
            python3 ./backend_setup/deploy.py
        else
            echo "No improvement -> no updates made"

        rm -rf ./question_answer/artifacts/answer
        fi
    fi
done
exit 0