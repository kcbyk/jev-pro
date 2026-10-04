#!/bin/bash
for pid in $(ps -eo pid,args | awk '$2 ~ /python3?$/ && $3 ~ /serve_pro/ {print $1}'); do
  kill "$pid" && echo "killed $pid"; sleep 1
done
