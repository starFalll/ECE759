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

cp common.h common_backup.h

for ((i = 1; i <= 16; i *= 2))
do
echo """#define FOR_SCHEDULE_TYPE $1

#define CHUNKS_PER_THREAD $i

#define OMP_SCHEDULE(type, chunk) schedule(type, chunk)""" > common.h

sh build.sh
echo "begin to write ${1}_${i}_threads_8_results.txt"
bash batchRun.sh ${2} | tee ../results/${1}_${i}_threads_8_results.txt

done

cp common_backup.h common.h
rm common_backup.h