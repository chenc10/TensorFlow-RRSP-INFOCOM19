import sys
f=open(sys.argv[1], 'r')
d=f.readlines()
f.close()
s = []
for i in range(len(d)):
	if 'blocked' in d[i]:
		tmp = d[i].split('blocked by:')[1]
		s.append(float(tmp))
print sum(s)/len(s)
