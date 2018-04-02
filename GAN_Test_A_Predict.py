from keras.layers import Input
from keras.models import load_model
from ipywidgets import IntProgress
import numpy as np
import Read_Data as RD
import GAN as gan
from numpy.linalg import cholesky
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
s_minority = np.dot(np.random.randn(sampleNo_minority, 2), R*0.4) + mu_minority

min_max_scaler = preprocessing.MinMaxScaler()
all_set = np.concatenate((o, s_minority))
min_max_scaler.fit(all_set)
o_trans = min_max_scaler.transform(o)
s_trans = min_max_scaler.transform(s_minority)

input_dim = 1
print('Load Model')
G_dense = 100
D_dense = 200
Pre_train_epoches = 100
Train_epoches = 300
Model_name = "G_Model_G-dense_" + str(G_dense) + "_pretrain_" + str(Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_maintrain_" + str(Train_epoches) + ".h5"
model = load_model(Model_name)

print('Generate Fake Samples')
Feature_samples = s_trans
print(Feature_samples[0])
print(Feature_samples[-1])

Num_Create_samples = 300
Noise_Input = np.random.uniform(0, 1, size=[Num_Create_samples, input_dim])
Sudo_Samples = model.predict(Noise_Input)


plt.plot(o_trans[:,0],o_trans[:,1])
plt.scatter(s_trans[:,0], s_trans[:,1], marker = '+', color = 'r', label='2', s = 3)
plt.scatter(Sudo_Samples[:,0], Sudo_Samples[:,1],marker = 'o', color = '#539caf', label='1', s = 3, alpha=0.3)
plt.show()


