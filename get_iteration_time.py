import sys
f=open(sys.argv[1], 'r')
d=f.readlines()
f.close()
s=[]
for i in range(len(d)):
	if 'sec/batch' in d[i]:
		tmp=d[i].split('sec/batch')[0].split(';')[-1]
		s.append(float(tmp))
print sum(s[10:])/len(s[10:])
