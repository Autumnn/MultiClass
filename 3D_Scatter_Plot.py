from __future__ import print_function
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "segment0.dat"

RD.Initialize_Data(dir)

for i in range(14, 19):
    for j in range(i+1, 19):
        for k in range(j+1, 19):
            print(i, j ,k)
            ax = plt.subplot(111, projection='3d')
            ax.scatter(RD.Negative_Feature[:,i], RD.Negative_Feature[:,j], RD.Negative_Feature[:,k], marker = 'o', color = '#539caf', label='1', s = 30, alpha=0.3)
            ax.scatter(RD.Positive_Feature[:,i], RD.Positive_Feature[:,j], RD.Positive_Feature[:,k], marker = '+', color = 'r', label='2', s = 50)
            plt.show()