
num=0
for i in $slaves
do
#	nohup ssh $i "stress-ng -c 8 -l 10" &
#	nohup ssh $i "python om.py 0.8" &
	nohup ssh $i "sudo iftop -t" > /home/ubuntu/iftop-slave$num.log 2>&1 &
	num=$((num+1))
done
