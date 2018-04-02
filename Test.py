import numpy as np
from numpy.linalg import cholesky
from ipywidgets import IntProgress
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.layers import Input

input_dim = 2
G_in = Input(shape=[input_dim])
print(G_in)
