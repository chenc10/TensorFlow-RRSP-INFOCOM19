ps_hosts=''
for i in $ps
do
	ps_hosts=$ps_hosts$i":22234,"
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

#ps_hosts='172.31.40.208:22234,172.31.45.168:22234'
num=0
for i in $ps
do
#nohup /home/ubuntu/chen-models/inception/bazel-bin/inception/imagenet_distributed_train --batch_size=32 --data_dir=/home/ubuntu/imagenet_data --job_name=ps --task_id=0 --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts  >/home/ubuntu/master1.log 2>&1 &
	nohup ssh $i "python /home/ubuntu/chen-models/inception/imagenet_inception_ssp.py --batch_size=32 --data_dir=/home/ubuntu/imagenet_data --job_name=ps --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts"  >/home/ubuntu/master$num.log 2>&1 &
	num=$((num+1))
done

num=0
for i in $slaves
do
#	nohup ssh $i "stress-ng -c 8 -l 10" &
#	nohup ssh $i "python om.py 0.8" &
	nohup ssh $i "python /home/ubuntu/chen-models/inception/imagenet_inception_ssp.py --batch_size=32 --data_dir=/home/ubuntu/imagenet_data --job_name=worker --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" > /home/ubuntu/slave$num.log 2>&1 &
#	ssh $i "~/tensorflow-inception/bazel-bin/inception/imagenet_distributed_train --batch_size=32 --data_dir=/home/ubuntu/imagenet_data --job_name=worker --task_id=$num --ps_hosts='172.31.45.172:22234' --worker_hosts=$worker_hosts" > slave$num.log 2>&1 
	num=$((num+1))
done
