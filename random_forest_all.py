# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:55:08 2019

@author: Vipin
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score ,precision_score
import seaborn as sns
fnames=['ddos','wednesday','infiltration','portscan','tuesday','webattack','friday']
features_imp=pd.DataFrame(columns=['Features','ddos','wednesday','infiltration','portscan','tuesday','webattack','friday'])
#fnames=['webattack']
output_values=pd.DataFrame(columns=['Features','ddos','wednesday','infiltration','portscan','tuesday','webattack','friday'])
for fname in fnames:
    
    
    data_total=pd.read_csv("Data/"+fname+".csv",sep=",")
    data_total.columns=data_total.columns.str.strip().str.replace(' ','_').str.replace("/s",'_per_sec').str.lower()

    data_total["flow_bytes_per_sec"]=data_total.flow_bytes_per_sec.astype(float)
    data_total["flow_packets_per_sec"]=data_total.flow_packets_per_sec.astype(float)
    data_total=data_total.replace(np.inf,np.nan)
    data_total=data_total.dropna()
    #Y=data_total['label']
    #Y=numeric.fit_transform(data_total['label'].astype('str'))
    
           
    data_attack=data_total[data_total.label!='BENIGN']
    data_normal=data_total[data_total.label=='BENIGN']
    label=data_total['label']
    Y,labels=label.factorize(sort=True)

    data_total=data_total.drop('label',axis=1)
    if(flag==1):
        features_imp['Features']=data_total.columns.values
        output_values['Features']=['Precision','Accuracy','F1_score']
        flag=0
    seed=123
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(data_total,Y,test_size=0.2,random_state=seed)
   
    model=RandomForestClassifier(n_estimators=10,random_state=seed)
    model.fit(Xtest,Ytest)
    feature_imp=model.feature_importances_
    features_imp[fname]=feature_imp
#print(clf.feature_importances_)
    Ypred=model.predict(Xtest)

    conf_mat=confusion_matrix(Ytest,Ypred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title("Confusion matrix "+fname+" attacks")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    
    precision=precision_score(Ytest,Ypred,average='micro')
    print("Precision is ",precision)
    f1=f1_score(Ytest,Ypred,average='micro')
    print("F1 Score ",f1)
    accuracy=accuracy_score(Ytest,Ypred)
    print("Accuracy  is ",accuracy)
    output=[precision,accuracy,f1]
    output_values[fname]=output  # Individual precision,accuracy and f1 score stored here
    

    #pickle.dump(model,open('portscan_rf.sav','wb'))
    #plt.show()
    plt.savefig(fname+"_rf.png")
 