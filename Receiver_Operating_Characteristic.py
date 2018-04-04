import math
import numpy as np
from sklearn import svm, preprocessing
import Read_Data as RD
from matplotlib import pyplot as plt

Num_Cross_Folders = 5
Gamma = np.arange(0.003, 0.021, 0.001)
G_Mean = np.linspace(0, 0, len(Gamma))
X_value = np.linspace(0, 0, len(Gamma))
Y_value = np.linspace(0, 0, len(Gamma))


for g in range(len(Gamma)):
    G_Mean_Temp = np.linspace(0, 0, Num_Cross_Folders)
    X_value_Temp = np.linspace(0, 0, Num_Cross_Folders)
    Y_value_Temp = np.linspace(0, 0, Num_Cross_Folders)
    for j in range(Num_Cross_Folders):
#        dir_train = "glass1-5-fold/glass1-5-" + str(j+1) + "tra.dat"
#        dir_test = "segment0-5-fold/segment0-5-" + str(j+1) + "tst.dat"
#        dir_train = "segment0-5-fold/segment0_SMOTE/result" + str(j) + "s0.tra"
#        dir_test = "segment0-5-fold/segment0_SMOTE/result" + str(j) + "s0.tst"
#        dir_train = "segment0-5-fold/segment0_SMOTE_TomekLinks/result" + str(j) + "s0.tra"
#        dir_test = "segment0-5-fold/segment0_SMOTE_TomekLinks/result" + str(j) + "s0.tst"
#         dir_train = "yeast4-5-fold/yeast4-5-" + str(j + 1) + "tra.dat"
        dir_test = "yeast4-5-fold/yeast4-5-" + str(j + 1) + "tst.dat"
#        dir_train = "yeast4-5-fold/yeast4_SMOTE/result" + str(j) + "s0.tra"
#        dir_test = "yeast4-5-fold/yeast4_SMOTE/result" + str(j) + "s0.tst"
#        dir_train = "yeast4-5-fold/yeast4_SMOTE_ENN/result" + str(j) + "s0.tra"
#        dir_test = "yeast4-5-fold/yeast4_SMOTE_ENN/result" + str(j) + "s0.tst
        dir_train = "yeast4-5-fold/yeast4_SMOTE_RSB/yeast4-5-" + str(j+1) + "tra.dat"


        RD.Initialize_Data(dir_train)
        Train_Feature_o = RD.get_feature()
        Train_Label = RD.get_label()
        Train_Label = Train_Label.ravel()

        RD.Initialize_Data(dir_test)
        Test_Feature_o = RD.get_feature()
        Test_Label = RD.get_label()
        Test_Label = Test_Label.ravel()

        min_max_scaler = preprocessing.MinMaxScaler()
        all_set = np.concatenate((Train_Feature_o, Test_Feature_o))
        min_max_scaler.fit(all_set)
        Train_Feature = min_max_scaler.transform(Train_Feature_o)
        Test_Feature = min_max_scaler.transform(Test_Feature_o)

        clf = svm.SVC(C=1, kernel='rbf', gamma=Gamma[g])
        clf.fit(Train_Feature, Train_Label)
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
        X_value_Temp[j] = FP / (TN + FP)
        Y_value_Temp[j] = TP / (TP + FN)
#        print('G_Mean =', G_Mean_Temp[j], 'Sensitivity = ', Sensitivity_Temp[j], 'Specificity = ', Specificity_Temp[j])

    G_Mean[g] = G_Mean_Temp.mean()
    X_value[g] = X_value_Temp.mean()
    Y_value[g] = Y_value_Temp.mean()
#    print("Gamma = ", Gamma[g], ", Average_G_Mean = ", G_Mean[g],
#          ", Average_Sensitivity = ", Sensitivity[g], ", Average_Specificity = ", Specificity[g])

print("G_Mean", G_Mean)
#print("Sensitivity", Sensitivity)
#print("Specificity", Specificity)

plt.plot(Gamma, G_Mean)
plt.show()

plt.scatter(X_value, Y_value)
plt.axis([0, 1, 0, 1])
plt.show()

