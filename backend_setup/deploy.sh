#!/bin/bash
while :
do
    git fetch
    if [ "$(git rev-list HEAD...origin/main --count)" != 0 ]; then
        echo "Updating backend"
        git pull
        python deploy.py
    fi
done

exit 0