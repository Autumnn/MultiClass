from keras.layers import Input
from ipywidgets import IntProgress
import Read_Data as RD
import GAN as gan

dir = "page-blocks0.dat"
RD.Initialize_Data(dir)

print('Generate Fake Samples')
Feature_samples = RD.get_positive_feature()
print(Feature_samples[0])
print(Feature_samples[-1])

#Fake_sample = gan.sample_data(Feature_samples)
#print(Fake_sample.shape)
#print(Fake_sample[0])
#print(Fake_sample[-1])

print('Generate Models')
G_in = Input(shape=[6])
G, G_out = gan.get_generative(G_in)
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

D_in = Input(shape=[10])
D, D_out = gan.get_discriminative(D_in)
D.summary()

GAN_in = Input([6])
GAN, GAN_out = gan.make_gan(GAN_in, G, D)
GAN.summary()

gan.pretrain(G, D, Feature_samples)

d_loss, g_loss = gan.train(GAN, G, D, Feature_samples, verbose=True)


