#!/bin/bash
while ! lt --port 11700 --subdomain admirer; do
    sleep 1
done