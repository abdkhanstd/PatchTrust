#!/bin/bash
cd ~/patchtrust/code
while pgrep -f "m patchtrust[.]train" >/dev/null; do sleep 60; done
while pgrep -f "jobs_eval_read[y]" >/dev/null; do sleep 30; done
scripts/run_queue.sh scripts/jobs_eval.txt 4 0,1,0,1
echo ALLDONE
