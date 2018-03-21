from __future__ import print_function
import numpy as np


#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "glass1.dat"

Num_lines = len(open(dir,'r').readlines())
data_info_lines = 0
num_columns = 0
with open(dir, "r") as get_info:
    print("name", get_info.name)
    for line in get_info:
        if line.find("@") == 0:
            data_info_lines += 1
        else:
            columns = line.split(",")
            num_columns = len(columns)
            break


Num_Samples = Num_lines - data_info_lines
print(Num_Samples)
Num_Features = num_columns - 1
Features = np.ones((Num_Samples, Num_Features))
Labels = np.ones((Num_Samples,1))

with open(dir, "r") as data_file:
    print("name", data_file.name)
    l = 0
    for line in data_file:
        l += 1
        if l > data_info_lines:
            #print(line)
            row = line.split(",")
            length_row = len(row)
            #print('Row length',length_row)
            #print(row[0])
            for i in range(length_row):
                if i < length_row - 1:
                    Features[l - data_info_lines - 1][i] = row[i]
                    #print(Features[l-14][i])
                else:
                    attri = row[i].strip()
                    #print(attri)
                    if attri == 'negative':
                        Labels[l - data_info_lines - 1][0]=0
                        #print(Labels[l-14][0])
                    else:
                        Labels[l - data_info_lines - 1][0]=1

print(Features[0])
for col in range(Num_Features):
    mean = np.mean(Features[:, col])
    Features[:, col] -= mean
print(Features[0])

Cov_Matrix = np.cov(Features.T)
for i in range(Num_Features):
    for value in Cov_Matrix[i]:
        print(value, " ", end='')
    print()

eigen_value, eigen_vector = np.linalg.eig(Cov_Matrix)
for v in eigen_value:
    print(v, ' ', end='')
