#!/bin/bash

# batch run program from thread 1 to max_thread_num

if [ $# -lt 1 ]; then
    echo "Usage: $0 <max_thread_num>"
    exit 1
fi

for ((i=1; i <= $1; i++))
do
    # echo "begin run ${i} threads"
    ./stitch_image $i
done