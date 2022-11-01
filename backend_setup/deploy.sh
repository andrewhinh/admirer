#!/bin/bash
while :
do
    git fetch
    if [ "$(git rev-list HEAD...origin/main --count)" != 0 ]; then
        echo "Updating backend"
        git pull
        jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=None --execute ./backend_setup/continuous_deployment.ipynb
    fi
done

exit 0