#!/bin/bash
until lt --port 11700 --subdomain admirer
do
  echo "Try again"
done