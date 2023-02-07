#!/bin/bash
while ! docker run -it -e NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN}" ngrok/ngrok http 11700 --subdomain=admirer; do
    sleep 1
done