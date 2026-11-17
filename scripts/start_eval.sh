#!/bin/bash
cd ~/patchtrust/code
: > scripts/jobs_eval_ready.txt
while IFS="|" read -r n m c; do
  ck=$(echo "$c" | grep -o "runs/[A-Za-z0-9_]*/best.pt" | head -1)
  d=$(dirname ~/patchtrust/$ck)
  [ -e $d/train_summary.json ] && echo "$n|$m|$c" >> scripts/jobs_eval_ready.txt
done < scripts/jobs_eval.txt
wc -l scripts/jobs_eval_ready.txt
nohup scripts/run_queue.sh scripts/jobs_eval_ready.txt 3 0,1,1 > ~/patchtrust/logs/queue_eval1.log 2>&1 < /dev/null &
nohup bash -c 'while pgrep -f patchtrust.train >/dev/null; do sleep 60; done; while pgrep -f jobs_eval_ready >/dev/null; do sleep 30; done; scripts/run_queue.sh scripts/jobs_eval.txt 4 0,1,0,1' > ~/patchtrust/logs/queue_eval2.log 2>&1 < /dev/null &
