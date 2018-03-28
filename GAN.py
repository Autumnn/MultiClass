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

def sample_data(samples):   # ACtually no use, just for test --> it can generate random fake samples
    Number_of_Feature = samples.shape[1]
#    Number_of_Samples = samples.shape[0]
    size = list(samples.shape)
#    print(size)
    Fake_Sample = np.random.rand(size[0],size[1])
    for i in range(Number_of_Feature):
        Min = min(samples[:,i])
        Dis = max(samples[:,i]) - Min
        print("i=", i, " Min=", Min, " Dis", Dis)
        Fake_Sample[:,i] = Fake_Sample[:,i] * Dis + Min
    return Fake_Sample

def sample_data_and_gen(G, samples, noise_dim = 6):
    #XT = sample_data(samples)
    XT = samples
    size = list(samples.shape)
    XN_noise = np.random.uniform(0, 1, size=[size[0], noise_dim])
    print("XN_noise:")
    print(XN_noise[0])
    print(XN_noise[-1])
    XN = G.predict(XN_noise)
    print("XN:")
    print(XN[0])
    print(XN[-1])
    X = np.concatenate((XT, XN))
    y = np.zeros((2*size[0], 2))
    y[:size[0], 0] = 1
    y[size[0]:, 1] = 1
    return X, y

def pretrain(G, D, samples, noise_dim = 6, batch_size=64):
    X, y = sample_data_and_gen(G, samples, noise_dim=noise_dim)
    set_trainability(D, True)
    D.fit(X, y, epochs=10, batch_size=batch_size)

#pretrain(G, D)