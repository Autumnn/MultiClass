import math
import numpy as np
from sklearn import svm
import Read_Data as RD
from matplotlib import pyplot as plt

Num_Cross_Folders = 5
Gamma = np.arange(0.15, 0.25, 0.05)
G_Mean = np.linspace(0, 0, len(Gamma))
Sensitivity = np.linspace(0, 0, len(Gamma))
Specificity = np.linspace(0, 0, len(Gamma))


for g in range(len(Gamma)):
    G_Mean_Temp = np.linspace(0, 0, Num_Cross_Folders)
    Sensitivity_Temp = np.linspace(0, 0, Num_Cross_Folders)
    Specificity_Temp = np.linspace(0, 0, Num_Cross_Folders)
    for j in range(Num_Cross_Folders):
#        dir_train = "glass1-5-fold/glass1-5-" + str(j+1) + "tra.dat"
#        dir_test = "glass1-5-fold/glass1-5-" + str(j+1) + "tst.dat"
        dir_train = "page-blocks0-5-fold/page-blocks0-5-" + str(j + 1) + "tra.dat"
        dir_test = "page-blocks0-5-fold/page-blocks0-5-" + str(j + 1) + "tst.dat"

        RD.Initialize_Data(dir_train)
        Train_Feature = RD.get_feature()
        Train_Label = RD.get_label()
        Train_Label = Train_Label.ravel()

        clf = svm.SVC(C=1, kernel='rbf', gamma=Gamma[g])
        clf.fit(Train_Feature, Train_Label)

        RD.Initialize_Data(dir_test)
        Test_Feature = RD.get_feature()
        Test_Label = RD.get_label()

        Test_Label = Test_Label.ravel()
        Labels_Predict = clf.predict(Test_Feature)
#        print(np.array([Test_Label, Labels_Predict]))
        TP = 0.0
        FP = 0.0
        FN = 0.0
        TN = 0.0
        l = len(Test_Label)
        for i in range(l):
            if Test_Label[i] == 0:
                if Labels_Predict[i] == 0:
                    TN += 1
                else:
                    FP += 1
            else:
                if Labels_Predict[i] == 1:
                    TP += 1
                else:
                    FN += 1

#        print('TP=', TP, 'FN', FN, 'FP', FP, 'TN', TN)
        G_Mean_Temp[j] = math.sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
        Sensitivity_Temp[j] = TP / (TP + FN)
        Specificity_Temp[j] = TN / (TN + FP)
        print('G_Mean =', G_Mean_Temp[j], 'Sensitivity = ', Sensitivity_Temp[j], 'Specificity = ', Specificity_Temp[j])

    G_Mean[g] = G_Mean_Temp.mean()
    Sensitivity[g] = Sensitivity_Temp.mean()
    Specificity[g] = Specificity_Temp.mean()
    if Gamma[g] == 0.2:
        FileWrite = "page-blocks0-5-fold/OverSampling_Test_GAN_with_or_without_SVM.txt"
        with open(FileWrite, 'a') as w:
            w.write("G_Mean: " + str(G_Mean_Temp) + "\n")
            w.write("Sensitivity: " + str(Sensitivity_Temp) + "\n")
            w.write("Specificity: " + str(Specificity_Temp) + "\n")
#    print("Gamma = ", Gamma[g], ", Average_G_Mean = ", G_Mean[g],
#          ", Average_Sensitivity = ", Sensitivity[g], ", Average_Specificity = ", Specificity[g])

#print("G_Mean", G_Mean)
#print("Sensitivity", Sensitivity)
#print("Specificity", Specificity)

plt.plot(Gamma, G_Mean)
plt.show()

plt.plot(Gamma, Sensitivity)
plt.show()

plt.plot(Gamma, Specificity)
plt.show()