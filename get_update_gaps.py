import sys
num = 16
update_times = []
for i in range(num):
	f = open(sys.argv[1] + '/slave'+str(i)+'.log', 'r')
	d = f.readlines()
	f.close()
	for i in range(len(d)):
		if 'loss =' in d[i]:
			tmp = d[i].split('time:')[1].split(';')[0]
			update_times.append(float(tmp))

update_times.sort()
gaps = []
for i in range(len(update_times)-1):
	gaps.append(update_times[i+1] - update_times[i])
gaps.sort()
print gaps

