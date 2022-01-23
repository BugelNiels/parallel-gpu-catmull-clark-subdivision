rm timings.txt
for run in {1..50}; do
	./subdivide $1 $2
	sleep 1
done