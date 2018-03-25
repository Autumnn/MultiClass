from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "segment0.dat"

RD.Initialize_Data(dir)

for i in range(16, 19):
    for j in range(16, 19):
        if i != j:
            fig = plt.figure()
            p1 = plt.hist2d(RD.Features[:,i], RD.Features[:,j], bins = 30)

            File_name = "2D_Distributions_of_" + str(i) + "_and_" + str(j) + "_Feature.png"
            fig.savefig(File_name)