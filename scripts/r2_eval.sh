#!/bin/bash
cd ~/patchtrust/code
mkdir -p ~/patchtrust/ksem
while ! grep -q R2TRAINDONE ~/patchtrust/logs/queue_r2.log; do sleep 60; done
scripts/run_queue.sh scripts/jobs_r2eval.txt 5 0,1,0,1,0
echo R2EVALDONE
