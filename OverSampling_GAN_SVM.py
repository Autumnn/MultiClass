from keras.layers import Input
from ipywidgets import IntProgress
import Read_Data as RD
import GAN as gan
import numpy as np
from sklearn import svm
import math

def GAN_Build(Feature_samples):
    print('Generate Models')
    G_in = Input(shape=[6])
    G, G_out = gan.get_generative(G_in)
    G.summary()

    D_in = Input(shape=[10])
    D, D_out = gan.get_discriminative(D_in)
    D.summary()

    GAN_in = Input([6])
    GAN, GAN_out = gan.make_gan(GAN_in, G, D)
    GAN.summary()

    gan.pretrain(G, D, Feature_samples)
    gan.train(GAN, G, D, Feature_samples, verbose=True)
    return G

def Over_Sampling(G, Num_samples, noise_dim):
    Noise_Input = np.random.uniform(0, 1, size=[Num_samples, noise_dim])
    Sudo_Samples = G.predict(Noise_Input)
    return Sudo_Samples

def Metric_Analysing(test, predict):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    l = len(test)
    for i in range(l):
        if test[i] == 0:
            if predict[i] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if predict[i] == 1:
                TP += 1
            else:
                FN += 1

    #        print('TP=', TP, 'FN', FN, 'FP', FP, 'TN', TN)
    G_Mean_Temp = math.sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    Sensitivity_Temp = TP / (TP + FN)
    Specificity_Temp = TN / (TN + FP)
    return G_Mean_Temp, Sensitivity_Temp, Specificity_Temp

Num_Cross_Folders = 5
G_Mean = np.linspace(0, 0, Num_Cross_Folders)
Sensitivity = np.linspace(0, 0, Num_Cross_Folders)
Specificity = np.linspace(0, 0, Num_Cross_Folders)
G_Mean_GAN = np.linspace(0, 0, Num_Cross_Folders)
Sensitivity_GAN = np.linspace(0, 0, Num_Cross_Folders)
Specificity_GAN = np.linspace(0, 0, Num_Cross_Folders)

for j in range(Num_Cross_Folders):
#        dir_train = "glass1-5-fold/glass1-5-" + str(j+1) + "tra.dat"
#        dir_test = "glass1-5-fold/glass1-5-" + str(j+1) + "tst.dat"
    dir_train = "page-blocks0-5-fold/page-blocks0-5-" + str(j+1) + "tra.dat"
    dir_test = "page-blocks0-5-fold/page-blocks0-5-" + str(j+1) + "tst.dat"

    RD.Initialize_Data(dir_train)
    Train_Feature = RD.get_feature()
    Train_Label = RD.get_label()
    Train_Label = Train_Label.ravel()
    print(Train_Feature.shape)
    print(Train_Label.size)

#    clf = svm.SVC(C=1, kernel='rbf', gamma= 0.2)
#    clf.fit(Train_Feature, Train_Label)

    Feature_samples = RD.get_positive_feature()
    G = GAN_Build(Feature_samples)
    Sudo_Samples = Over_Sampling(G, RD.Num_negative - RD.Num_positive, 6)
    print(Sudo_Samples[0])
    print(Sudo_Samples[-1])
    Train_Feature = np.concatenate((Train_Feature, Sudo_Samples))
    print(Train_Feature[0])
    print(Train_Feature[-1])
    Augment_Label = np.ones((RD.Num_negative - RD.Num_positive),dtype=np.int16)
    Train_Label = np.append(Train_Label, Augment_Label)
    print(Train_Feature.shape)
    print(Train_Label.size)

    clf_gan = svm.SVC(C=1, kernel='rbf', gamma= 0.2)
    clf_gan.fit(Train_Feature, Train_Label)

    RD.Initialize_Data(dir_test)
    Test_Feature = RD.get_feature()
    Test_Label = RD.get_label()

    Test_Label = Test_Label.ravel()
#    Labels_Predict = clf.predict(Test_Feature)
#    G_Mean[j], Sensitivity[j], Specificity[j] = Metric_Analysing(Test_Label, Labels_Predict)

    Labels_Predict_GAN = clf_gan.predict(Test_Feature)
    G_Mean_GAN[j], Sensitivity_GAN[j], Specificity_GAN[j] = Metric_Analysing(Test_Label, Labels_Predict_GAN)

FileWrite = "page-blocks0-5-fold/OverSampling_Test_GAN_with_or_without_SVM.txt"
with open(FileWrite, 'a') as w:
#    w.write("G_Mean: " + str(G_Mean) + "\n")
#    w.write("Sensitivity: " + str(Sensitivity) + "\n")
#    w.write("Specificity: " + str(Specificity) + "\n")
    w.write("G_Mean_GAN: " + str(G_Mean_GAN) + "\n")
    w.write("Sensitivity_GAN: " + str(Sensitivity_GAN) + "\n")
    w.write("Specificity_GAN: " + str(Specificity_GAN) + "\n")




