from keras.layers import Input
from ipywidgets import IntProgress
import numpy as np
import Read_Data as RD
import GAN as gan
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

sampleNo = 1000
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = cholesky(Sigma)
s = (np.dot(np.random.randn(sampleNo, 2), R) + mu)/10

print('Generate Fake Samples')
Feature_samples = s
print(Feature_samples[0])
print(Feature_samples[-1])

#Fake_sample = gan.sample_data(Feature_samples)
#print(Fake_sample.shape)
#print(Fake_sample[0])
#print(Fake_sample[-1])

print('Generate Models')
G_in = Input(shape=[2])
G, G_out = gan.get_generative(G_in, out_dim=2)
G.summary()

#Train_Sample, Train_Label = gan.sample_data_and_gen(G, Feature_samples, noise_dim=6)
#print("Train_Sample:")
#print(Train_Sample[0])
#print(Train_Sample[RD.Num_positive-1])
#print(Train_Sample[RD.Num_positive])
#print(Train_Sample[-1])
#print("Train_Label:")
#print(Train_Label[0])
#print(Train_Label[RD.Num_positive-1])
#print(Train_Label[RD.Num_positive])
#print(Train_Label[-1])

D_in = Input(shape=[2])
D, D_out = gan.get_discriminative(D_in)
D.summary()

GAN_in = Input([2])
GAN, GAN_out = gan.make_gan(GAN_in, G, D)
GAN.summary()

gan.pretrain(G, D, Feature_samples, noise_dim=2)

d_loss, g_loss = gan.train(GAN, G, D, Feature_samples, noise_dim=2, verbose=True)

Noise_Input = np.random.uniform(0, 1, size=[sampleNo, 2])
Sudo_Samples = G.predict(Noise_Input)

plt.subplot(1,2,1)
#plt.scatter(Noise_Input[:,0],Noise_Input[:,1],marker = 'o', color = '#539caf', label='1', s = 3, alpha=0.3)
plt.scatter(Feature_samples[:,0], Feature_samples[:,1],marker = 'o', color = '#539caf', label='1', s = 3, alpha=0.3)

plt.subplot(1,2,2)
plt.scatter(Sudo_Samples[:,0], Sudo_Samples[:,1], marker = '+', color = 'r', label='2', s = 10)

plt.show()

