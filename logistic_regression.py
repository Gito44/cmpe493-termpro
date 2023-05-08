import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

base_directory1=r"D:\kod\term pro\cosine_similarities\problem00002_cosine_similarity.csv"
base_directory2=r"D:\kod\term pro\cosine_similarities\problem00004_cosine_similarity.csv"
base_directory3=r"D:\kod\term pro\cosine_similarities\problem00001_cosine_similarity.csv"
base_directory4=r"D:\kod\term pro\cosine_similarities\problem00003_cosine_similarity.csv"
base_directory5=r"D:\kod\term pro\cosine_similarities\problem00005_cosine_similarity.csv"
dataset1 = pd.read_csv(base_directory1,index_col=0)
dataset2 = pd.read_csv(base_directory2,index_col=0)
dataset3 = pd.read_csv(base_directory3,index_col=0)
dataset4 = pd.read_csv(base_directory4,index_col=0)
dataset5 = pd.read_csv(base_directory5,index_col=0)
data_final=pd.concat([dataset1,dataset2])
test_final=pd.concat([dataset3,dataset4,dataset5])

Xtrain = data_final.iloc[:, :-1].values
ytrain = data_final.iloc[:, -1].values
Xtest = test_final.iloc[:, :-1].values
ytest = test_final.iloc[:, -1].values



classifier = LogisticRegression(random_state = 0,multi_class='multinomial')
classifier.fit(Xtrain, ytrain)


classifiernb = GaussianNB()
classifiernb.fit(Xtrain, ytrain)


y_pred = classifier.predict(Xtest)
y_prednb = classifiernb.predict(Xtest)


lsuccess=0
lfail=0

nbsuccess=0
nbfail=0
for i in range(len(y_prednb)):
    if y_prednb[i]==ytest[i]:
        nbsuccess+=1
    else:
        nbfail+=1

for i in range(len(y_pred)):
    if y_pred[i] == ytest[i]:
        lsuccess += 1
    else:
        lfail += 1

#print(np.concatenate((y_prednb.reshape(len(y_prednb),1), y4.reshape(len(y4),1)),1))
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y4.reshape(len(y4),1)),1))
print(nbsuccess/(nbsuccess+nbfail))
print(lsuccess/(lsuccess+lfail))