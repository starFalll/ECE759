#!/bin/bash

# for each schedule type, run from chunk 1,2,4...,16, and save the results

if [ $# -lt 2 ]; then
    echo "Usage: $0 <schedule_type>(static|dynamic|guided) <max_thread_num>"
    exit 1
fi

if [[ "$1" != "static" && "$1" != "dynamic" && "$1" != "guided" ]]; then
    echo "Error: Invalid schedule type. Must be one of: static, dynamic, guided."
    exit 1
fi

sh build.sh

for ((i = 1; i <= 16; i *= 2))
do
export OMP_SCHEDULE="${1},${i}"

echo "begin to write ${1}_${i}_threads_8_results.txt"
bash batchRun.sh ${2} | tee ../results/${1}_${i}_threads_8_results.txt

done