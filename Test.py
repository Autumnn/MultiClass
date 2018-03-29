import numpy as np

a = np.ones((10),dtype=np.int16)
print(a)
b = np.array([[1,0,1],[0,1,0]])
print(b)
b = b.ravel()
print(b)
b = np.append(b, a)
print(b)

FileWrite = "page-blocks0-5-fold/OverSampling_Test_GAN_with_or_without_SVM.txt"
with open(FileWrite, 'a') as w:
    w.write("G_Mean: ")
    w.write(str(a) + "\n")
    w.write("Sensitivity: ")
    w.write(str(b))