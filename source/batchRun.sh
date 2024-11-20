for ((i=1; i <= 8; i++))
do
    # echo "begin run ${i} threads"
    ./stitch_image $i
done