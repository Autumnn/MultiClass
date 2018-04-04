from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "segment0.dat"

RD.Initialize_Data(dir)



for i in range(0, 19):
    for j in range(i+1, 19):
        if i != j:
            fig = plt.figure()
            p1 = plt.scatter(RD.Negative_Feature[:,i], RD.Negative_Feature[:,j], marker = 'o', color = '#539caf', label='1', s = 10, alpha=0.3)
            p2 = plt.scatter(RD.Positive_Feature[:,i], RD.Positive_Feature[:,j], marker = '+', color = 'r', label='2', s = 50)
            File_name = "Scatter_Plot_of_" + str(i) + "_and_" + str(j) + "_Feature.png"
            fig.savefig(File_name)


