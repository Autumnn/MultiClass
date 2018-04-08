import numpy as np
from numpy.linalg import cholesky
from sklearn import preprocessing
import matplotlib.pyplot as plt

a = np.random.randint(1,11,10)
for i in  range(5):
    a = np.vstack((a, np.random.randint(1,11,10)))
print(a)

#b_t = sum(a[0,:])
b_t = a[0,:] / float(sum(a[:,0]))
print(b_t)

b = preprocessing.normalize(a, axis=0, norm='l1')
print(b[0])

c_t = a[0,0]/sum(a[0,:])
print(c_t)

c = preprocessing.normalize(a, axis=1, norm='l1')
print(c[0])