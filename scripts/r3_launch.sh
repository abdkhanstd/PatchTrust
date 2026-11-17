#!/bin/bash
cd ~/patchtrust/code
mkdir -p ~/patchtrust/decomp
scripts/run_queue.sh scripts/jobs_r3a.txt 6 0,1,0,1,0,1
scripts/run_queue.sh scripts/jobs_r3b.txt 6 0,1,0,1,0,1
scripts/run_queue.sh scripts/jobs_r3c.txt 6 0,1,0,1,0,1
echo R3DONE
