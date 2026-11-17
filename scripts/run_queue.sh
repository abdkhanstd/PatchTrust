#!/bin/bash
# Usage: run_queue.sh jobs.txt NPAR GPU_LIST(comma)  -- each line of jobs.txt: NAME|ARGS for python -m patchtrust.<module>
# Jobs whose output marker exists are skipped, so the queue can be re-run after a connection drop.
JOBS=$1; NPAR=${2:-3}; GPUS=${3:-0}
cd ~/patchtrust/code
IFS=',' read -ra G <<< "$GPUS"
i=0
while IFS='|' read -r name marker cmd; do
  [ -z "$name" ] && continue
  if [ -e "$marker" ]; then echo "skip $name"; continue; fi
  gpu=${G[$((i % ${#G[@]}))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu ~/patchtrust/env/bin/python -m $cmd > ~/patchtrust/logs/$name.log 2>&1; echo "done $name $?" >> ~/patchtrust/logs/queue_done.txt ) &
  while [ $(jobs -rp | wc -l) -ge $NPAR ]; do sleep 5; done
done < $JOBS
wait
