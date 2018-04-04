from __future__ import print_function
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "yeast4.dat"

RD.Initialize_Data(dir)

#Features_Attribute = [0, 13 , 14, 15, 16, 17, 18]
Features_Attribute = np.arange(0, RD.Num_Features, 1)

l = len(Features_Attribute)

for i in range(0, l):
    for j in range(i+1, l):
        for k in range(j+1, l):
            X_index = Features_Attribute[i]
            Y_index = Features_Attribute[j]
            Z_index = Features_Attribute[k]
            print(X_index, Y_index, Z_index)
            ax = plt.subplot(111, projection='3d')
            ax.scatter(RD.Negative_Feature[:,X_index], RD.Negative_Feature[:,Y_index], RD.Negative_Feature[:,Z_index], marker = 'o', color = '#539caf', label='1', s = 30, alpha=0.3)
            ax.scatter(RD.Positive_Feature[:,X_index], RD.Positive_Feature[:,Y_index], RD.Positive_Feature[:,Z_index], marker = '+', color = 'r', label='2', s = 50)
            plt.show()