from __future__ import print_function
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "page-blocks0.dat"

RD.Initialize_Data(dir)

ax = plt.subplot(111, projection='3d')
ax.scatter(RD.Negative_Feature[:,4], RD.Negative_Feature[:,5], RD.Negative_Feature[:,3], marker = 'o', color = '#539caf', label='1', s = 10, alpha=0.3)
ax.scatter(RD.Positive_Feature[:,4], RD.Positive_Feature[:,5], RD.Positive_Feature[:,3], marker = '+', color = 'r', label='2', s = 50)

plt.show()