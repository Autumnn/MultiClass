from keras.layers import Input
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
s_minority = np.dot(np.random.randn(sampleNo_minority, 2), R*0.5) + mu_minority

min_max_scaler = preprocessing.MinMaxScaler()
all_set = np.concatenate((o, s_minority))
min_max_scaler.fit(all_set)
o_trans = min_max_scaler.transform(o)
s_trans = min_max_scaler.transform(s_minority)

input_dim = 2

print('Generate Fake Samples')
Feature_samples = s_trans
print(Feature_samples[0])
print(Feature_samples[-1])

print('Generate Models')
G_in = Input(shape=[input_dim])
G, G_out = gan.get_generative(G_in, out_dim=2)
G.summary()
D_in = Input(shape=[2])
D, D_out = gan.get_discriminative(D_in)
D.summary()
GAN_in = Input([input_dim])
GAN, GAN_out = gan.make_gan(GAN_in, G, D)
GAN.summary()

Pre_train_epoches = 1000
Train_epoches = 1000
gan.pretrain(G, D, Feature_samples, noise_dim=input_dim, epoches=Pre_train_epoches)
d_loss, g_loss = gan.train(GAN, G, D, Feature_samples, epochs= Train_epoches , noise_dim=input_dim, verbose=True)
Model_name = "Generator_Model_pretrain_" + str(Pre_train_epoches) + "_maintrain_" + str(Train_epoches) + ".h5"
G.save(Model_name)




