rm -f timings.txt
for run in {1..2}; do
	./subdivide $1 $2
	sleep 1
done