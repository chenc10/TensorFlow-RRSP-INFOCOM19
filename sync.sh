if [ ! -n "$1" ] ;then
	echo "no input"
	exit
fi
for s in $ps_other $slaves
do
	echo "- sudo rm -rf $s:`pwd`/$1"
	ssh $s "sudo rm -rf `pwd`/$1"
	echo "- scp $1 $s:`pwd`/$1"
	scp -r $1 $s:`pwd`/$1
	echo "finish "$s
	echo 
done
