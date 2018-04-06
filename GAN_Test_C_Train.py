from keras.layers import Input
import numpy as np
import GAN as gan
from numpy.linalg import cholesky
from sklearn import preprocessing


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

FileWrite = "Generated_minority.npy"
np.save(FileWrite, s_trans)
File_Majority = "Generated_majority.npy"
np.save(File_Majority, o_trans)

input_dim = 2

print('Generate Fake Samples')
Feature_samples = s_trans
print(Feature_samples[0])
print(Feature_samples[-1])

G_dense = 300
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
Train_epoches = 50000
gan.pretrain(G, D, Feature_samples, noise_dim=input_dim, epoches=Pre_train_epoches)
d_loss, g_loss = gan.train(GAN, G, D,Feature_samples, epochs= Train_epoches , noise_dim=input_dim, verbose=True)
Model_name = "GAN_C_G-dense_" + str(G_dense) + "_pretrain_" + str(Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_maintrain_" + str(Train_epoches) + ".h5"
G.save(Model_name)




