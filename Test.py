import numpy as np
from numpy.linalg import cholesky
from sklearn import preprocessing
import matplotlib.pyplot as plt

sampleNo = 800
mu = np.array([[1, 3]])
Sigma = np.array([[2, 2], [1, 1]])
print(Sigma)
R = cholesky(Sigma)
print(R)
o = (np.dot(np.random.randn(sampleNo, 2), R) + mu)

sampleNo_minority = 100
mu_minority = np.array([[9, 3]])
Sigma_minority = np.array([[1, 1.5], [0.5, 3]])
R = cholesky(Sigma_minority)
print(R)
s_minority = np.dot(np.random.randn(sampleNo_minority, 2), R*0.4) + mu_minority

min_max_scaler = preprocessing.MinMaxScaler()
all_set = np.concatenate((o, s_minority))
min_max_scaler.fit(all_set)
o_trans = min_max_scaler.transform(o)
s_trans = min_max_scaler.transform(s_minority)
o_size = list(o_trans.shape)

condition_samples = np.zeros((sampleNo_minority, 2))
for k_n in range(sampleNo_minority):
    min_dis = 100
    for k_o in range(o_size[0]):
        dis = np.linalg.norm(s_trans[k_n,:]-o_trans[k_o,:])
        if dis <= min_dis:
            min_dis = dis
            condition_samples[k_n,:] = o_trans[k_o,:]

d_write = np.hstack((s_trans, condition_samples))

[s_t, c_s] = np.hsplit(d_write, 2)

plt.scatter(o_trans[:,0],o_trans[:,1], color = '#539caf', alpha = 0.3)
plt.scatter(s_t[:,0], s_t[:,1], marker = '+', color = 'r', label='2', s = 3)
plt.scatter(c_s[:,0], c_s[:,1],marker = 'o', color = 'g', label='1', s = 3, alpha=1)
plt.show()