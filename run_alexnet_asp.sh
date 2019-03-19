ps_hosts=''
for i in $ps
do
	ps_hosts=$ps_hosts$i":22233,"
done
ps_hosts=${ps_hosts%%,}
echo $ps_hosts

worker_hosts=''
for i in $slaves
do
	worker_hosts=$worker_hosts$i":22234,"
done
worker_hosts=${worker_hosts%%,}
echo $worker_hosts

num=0
for i in $ps
do
	nohup ssh $i "python /home/ubuntu/chen-models/alexnet/cifar10_alexnet_asp.py --job_name=ps --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts"  >/home/ubuntu/master$num.log 2>&1 &
	num=$((num+1))
done

num=0
for i in $slaves
do
#	nohup ssh $i "stress-ng -c 8 -l 10" &
#	nohup ssh $i "python om.py 0.8" &
	nohup ssh $i "python /home/ubuntu/chen-models/alexnet/cifar10_alexnet_asp.py --job_name=worker --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" > /home/ubuntu/slave$num.log 2>&1 &
	num=$((num+1))
done
