from keras import optimizers
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import numpy as np
import Read_Data as RD
import GAN as gan


dir = "page-blocks0.dat"
RD.Initialize_Data(dir)

print('Generate Fake Samples')
Feature_samples = RD.get_positive_feature()
Fake_sample = gan.sample_data(Feature_samples)

print('Generate Models')
G_in = Input(shape=[6])
G, G_out = gan.get_generative(G_in)
G.summary()
D_in = Input(shape=[10])
D, D_out = gan.get_discriminative(D_in)
D.summary()
gan.pretrain(G, D, Feature_samples)


