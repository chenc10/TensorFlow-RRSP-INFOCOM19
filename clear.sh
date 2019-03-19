for i in $ps
do 
	ssh $i "sudo pkill python"
	echo "finish clear - "$i
done
for i in $slaves
do 
	ssh $i "sudo pkill python"
	echo "finish clear - "$i
done
