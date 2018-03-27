from keras import optimizers
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import numpy as np

def get_generative(G_in, dense_dim=200, out_dim=10, lr=1e-3):
    x = Dense(dense_dim)(G_in)
    x = Activation('tanh')(x)
    G_out = Dense(out_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

#G_in = Input(shape=[6])
#G, G_out = get_generative(G_in)
#G.summary()

def get_discriminative(D_in, dense_dim = 200, lr=1e-3):
    x = Dense(dense_dim)(D_in)
    x = Activation('sigmoid')(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

#D_in = Input(shape=[10])
#D, D_out = get_discriminative(D_in)
#D.summary()


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_in, G, D):
    set_trainability(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out


#GAN_in = Input([10])
#GAN, GAN_out = make_gan(GAN_in, G, D)
#GAN.summary()

def sample_data(samples):
    Number_of_Feature = samples.shape[1]
#    Number_of_Samples = samples.shape[0]
    Fake_Sample = np.random.rand(samples.shape)
    for i in range(Number_of_Feature):
        Min = min(samples[:,i])
        Dis = max(samples[:,i]) - Min
        Fake_Sample[:,i] = Fake_Sample[:,i] * Dis + Min
    return Fake_Sample

def sample_data_and_gen(G, noise_dim=10, n_samples=10000):
    XT = sample_data(n_samples=n_samples)
    XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    XN = G.predict(XN_noise)
    X = np.concatenate((XT, XN))
    y = np.zeros((2*n_samples, 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, y

def pretrain(G, D, noise_dim=10, n_samples=10000, batch_size=32):
    X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
    set_trainability(D, True)
    D.fit(X, y, epochs=1, batch_size=batch_size)

#pretrain(G, D)