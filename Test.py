import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

sampleNo = 1000
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = cholesky(Sigma)
s = (np.dot(np.random.randn(sampleNo, 2), R) + mu)/10
#plt.subplot(144)

Sudo_Samples = np.random.randn(sampleNo, 2)

plt.subplot(2,1,1)
plt.scatter(s[:,0],s[:,1],marker = 'o', color = '#539caf', label='1', s = 3, alpha=0.3)

plt.subplot(2,1,2)
plt.scatter(Sudo_Samples[:,0], Sudo_Samples[:,1], marker = '+', color = 'r', label='2', s = 3)

plt.show()
