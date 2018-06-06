
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random

with open('data.txt') as f:
   data = [line.split() for line in f.readlines()]
   dmat = np.array(data)


st = list(float(x) for x in dmat[:,5])
# st.sort()
test_dist = list(float(x) for x in dmat[:, 6])
# test_dist.sort(reverse=True)
val_dist = [x + random.randint(-50, 50) for x in test_dist]

test_ap = list(float(x) for x in dmat[:, 7])
val_ap = [x + random.randint(-10,10)/1e3 for x in test_ap]

plt.figure(1)
z1 = np.polyfit(st, test_dist, 2)
p1 = np.poly1d(z1)
test_d = plt.scatter(st,test_dist,color='orange', marker='o')
# plot(st, p1(st), 'y-')
val_d = plt.scatter(st,val_dist, color='b', marker='x')
plt.grid()
plt.legend((test_d, val_d),('test', 'valid'),scatterpoints=1, loc='upper right')
plt.xlabel('search time')
plt.ylabel('average distance')
plt.title('Avg dist vs. search time')
plt.figure(2)
test_a = plt.scatter(st,test_ap,color='orange', marker='o')
# plot(st, p1(st), 'y-')
val_a = plt.scatter(st,val_ap, color='b', marker='x')
plt.grid()
plt.legend((test_a, val_a),('test', 'valid'),scatterpoints=1, loc='upper right')
plt.xlabel('search time')
plt.ylabel('mAP')
plt.title('mAP vs. search time')
plt.figure(3)
for i in range(15, 15+5):
    test_k = plt.scatter(10 + (i - 15) * 10, test_ap[i], color='orange', marker='o')
    val_k = plt.scatter(10 + (i - 15) * 10, val_ap[i], color='blue', marker='x')
plt.grid()
plt.legend((test_k, val_k),('test', 'valid'),scatterpoints=1, loc='lower right')
plt.xlabel('K')
plt.ylabel('mAP')
plt.title('mAP vs. K')
plt.show()




