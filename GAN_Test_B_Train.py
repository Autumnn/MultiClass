from keras.layers import Input
from ipywidgets import IntProgress
import numpy as np
import Read_Data as RD
import GAN as gan
from numpy.linalg import cholesky
from sklearn import preprocessing
import matplotlib.pyplot as plt

sampleNo_minority = 100
mu_minority = np.array([[9, 3]])
Sigma_minority = np.array([[1, 1.5], [0.5, 3]])
R = cholesky(Sigma_minority)
print(R)
s_minority = np.dot(np.random.randn(sampleNo_minority, 2), R*0.4) + mu_minority

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(s_minority)
s_trans = min_max_scaler.transform(s_minority)

input_dim = 2

print('Generate Fake Samples')
Feature_samples = s_trans
print(Feature_samples[0])
print(Feature_samples[-1])

G_dense = 200
D_dense = 200
print('Generate Models')
G_in = Input(shape=[input_dim])
G, G_out = gan.get_generative(G_in, dense_dim=G_dense, out_dim=2)
G.summary()
D_in = Input(shape=[2])
D, D_out = gan.get_discriminative(D_in, dense_dim=D_dense)
D.summary()
GAN_in = Input([input_dim])
GAN, GAN_out = gan.make_gan(GAN_in, G, D)
GAN.summary()

Pre_train_epoches = 100
Train_epoches = 5000
gan.pretrain(G, D, Feature_samples, noise_dim=input_dim, epoches=Pre_train_epoches)
d_loss, g_loss = gan.train(GAN, G, D, Feature_samples, epochs= Train_epoches , noise_dim=input_dim, verbose=True)
Model_name = "GAN_G-dense_" + str(G_dense) + "_pretrain_" + str(Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_maintrain_" + str(Train_epoches) + ".h5"
G.save(Model_name)




