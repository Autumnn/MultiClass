import os
import math
import numpy as np
from sklearn import svm

G_Mean = np.array([0.0,0.0,0.0,0.0,0.0])
for j in range(5):
    dir_train = "glass1-5-fold/glass1-5-" + str(j+1) + "tra.dat"
    dir_test = "glass1-5-fold/glass1-5-" + str(j+1) + "tst.dat"
    #dir_train = "glass1/result" + str(j) + "s0.tra"
    #dir_test = "glass1/result" + str(j) + "s0.tst"

    Num_lines = len(open(dir_train,'r').readlines())
    Num_Samples = Num_lines-14
    print(Num_Samples)

    Features = np.ones((Num_Samples,9))
    Labels = np.ones((Num_Samples,1))

    with open(dir_train, "r") as data_file:
        print("name", data_file.name)
        l = 0
        for line in data_file:
            l += 1
            if l >= 15:
                #print(line)
                row = line.split(",")
                length_row = len(row)
                #print('Row length',length_row)
                #print(row[0])
                for i in range(length_row):
                    if i < length_row - 1:
                        Features[l-15][i] = row[i]
                        #print(Features[l-14][i])
                    else:
                        attri = row[i].strip()
                        #print(attri)
                        if attri == 'negative':
                            Labels[l-15][0]=0
                            #print(Labels[l-14][0])
                        else:
                            Labels[l-15][0]=1

    #print(Features)
    Labels = Labels.ravel()
    #print(Labels)

    clf = svm.SVC()
    clf.fit(Features, Labels)

    #SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr',degree=3,
    #    gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None,
    #    shrinking=True,tol=0.001, verbose=False)

    Num_lines_Test = len(open(dir_test,'r').readlines())
    Num_Samples_Test = Num_lines_Test-14
    print(Num_Samples_Test)

    Features_Test = np.ones((Num_Samples_Test,9))
    Labels_Test = np.ones((Num_Samples_Test,1))

    with open(dir_test, "r") as data_file_Test:
        print("name", data_file_Test.name)
        l = 0
        for line in data_file_Test:
            l += 1
            if l >= 15:
                #print(line)
                row = line.split(",")
                length_row = len(row)
                #print('Row length',length_row)
                #print(row[0])
                for i in range(length_row):
                    if i < length_row - 1:
                        Features_Test[l-15][i] = row[i]
                        #print(Features[l-14][i])
                    else:
                        attri = row[i].strip()
                        #print(attri)
                        if attri == 'negative':
                            Labels_Test[l-15][0]=0
                            #print(Labels[l-14][0])
                        else:
                            Labels_Test[l-15][0]=1

    Labels_Test = Labels_Test.ravel()
    Labels_Predict = clf.predict(Features_Test)
    print(np.array([Labels_Test,Labels_Predict]))
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    l = len(Labels_Test)
    for i in range(l):
        if Labels_Test[i] == 0:
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