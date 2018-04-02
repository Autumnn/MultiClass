import numpy as np
from numpy.linalg import cholesky
from ipywidgets import IntProgress
import matplotlib.pyplot as plt
from sklearn import preprocessing

t=[i*np.pi/180 for i in np.arange(0, 360)]
x=4*np.cos(t)
y=np.sin(t)
x_1=np.cos(np.pi/6)*x-np.sin(np.pi/6)*y + 4
y_1=np.cos(np.pi/6)*y+np.sin(np.pi/6)*x + 3
o = np.array([x_1,y_1])
o = o.transpose()

sampleNo_minority = 100
mu_minority = np.array([[9, 3]])
Sigma_minority = np.array([[1, 1.5], [0.5, 3]])
R = cholesky(Sigma_minority)
print(R)
s_minority = np.dot(np.random.randn(sampleNo_minority, 2), R*0.5) + mu_minority

min_max_scaler = preprocessing.MinMaxScaler()
all_set = np.concatenate((o, s_minority))
min_max_scaler.fit(all_set)
o_trans = min_max_scaler.transform(o)
s_trans = min_max_scaler.transform(s_minority)

plt.plot(o_trans[:,0],o_trans[:,1])
plt.scatter(s_trans[:,0], s_trans[:,1], marker = '+', color = 'r', label='2', s = 3)
plt.show()

