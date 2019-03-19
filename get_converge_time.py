import sys
loss_list=[]
f=open(sys.argv[1],'r')
d=f.readlines()
f.close()
for j in range(len(d)):
#    print d[j]
    if 'examples/sec' not in d[j]:
        continue
    if 'step' not in d[j]:
        continue
    if 'tensorflow' not in d[j]:
        continue
    tmp = d[j].split('loss = ')
#    print tmp
    time_str = tmp[1].split('(')[0]
    loss_list.append(float(time_str))
#print loss_list
for i in range(10, len(loss_list)):
    sign = True 
    for j in range(i-10, i):
        if loss_list[j] > 1.0: 
            sign = False
            break
    if sign == True:
        print i
        break

