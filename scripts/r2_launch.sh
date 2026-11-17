#!/bin/bash
cd ~/patchtrust/code
scripts/run_queue.sh scripts/jobs_r2a.txt 6 0,1,0,1,0,1
scripts/run_queue.sh scripts/jobs_r2b.txt 6 0,1,0,1,0,1
echo R2TRAINDONE
