#!/bin/bash
while :
do
    git fetch
    if [ "$(git rev-list HEAD...origin/main --count)" != 0 ]; then
        echo "Checking if backend needs to be updated..."
        git pull

        cd ..
        python ./training/stage_model.py --fetch

        BEST_ACC=$(< ./question_answer/evaluation/best_pica_acc.txt)
        NEW_ACC=$(python ./question_answer/evaluation/evalute_pica.py)
        if [ "$NEW_ACC" \> "$BEST_ACC" ]; then
            echo "Updating backend..."
            echo "$NEW_ACC" >| ./question_answer/evaluation/best_pica_acc.txt
            cd backend_setup || exit
            python deploy.py
        else
            echo "No improvement -> no updates made"
            cd backend_setup || exit

        rm -rf ./question_answer/artifacts
        rm -rf ./data
        fi
    fi
done
exit 0