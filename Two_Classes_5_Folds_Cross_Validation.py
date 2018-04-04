import os
import math
import numpy as np
from sklearn import svm, preprocessing
import Read_Data as RD

Num_Cross_Folders = 5
G_Mean = np.linspace(0, 0, Num_Cross_Folders)
for j in range(Num_Cross_Folders):
    dir_train = "segment0-5-fold/segment0-5-" + str(j+1) + "tra.dat"
    dir_test = "segment0-5-fold/segment0-5-" + str(j+1) + "tst.dat"
    #dir_train = "glass1/result" + str(j) + "s0.tra"
    #dir_test = "glass1/result" + str(j) + "s0.tst"

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

    clf = svm.SVC(C=1, kernel='rbf', gamma=0.2)
    clf.fit(Train_Feature, Train_Label)
    Labels_Predict = clf.predict(Test_Feature)

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

    print('TP=', TP, 'FN', FN, 'FP', FP, 'TN', TN)
    G_Mean[j] = math.sqrt((TP/(TP+FN))*(TN/(TN+FP)))
    print('G_Mean =', G_Mean[j])


print(G_Mean.mean())