
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score ,precision_score, precision_recall_curve,recall_score, classification_report, auc, roc_curve
flag=1
from keras import utils as utility
import seaborn as sns
fnames=['ddos','wednesday','infiltration','portscan','tuesday','webattack','friday']

#fnames=['webattack']
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
    label_count=np.unique(Y)
    Y=utility.to_categorical(Y,label_count.shape[0])

    data_total=data_total.drop('label',axis=1)
    if (label_count.shape[0] > 2):
        loss='categorical_crossentropy'
    else:
        loss='binary_crossentropy'
   
    seed=123
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(data_total,Y,test_size=0.2,random_state=seed)
   
    ae_epoch=25
    input_dim=Xtrain.shape[1]
    encoding_dim = 20
    hidden_dim = int(encoding_dim / 4)
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    encoder = Dense(hidden_dim, activation="sigmoid")(encoder)
    decoder = Dense(hidden_dim, activation='relu')(encoder)
    decoder1 = Dense(input_dim, activation='sigmoid')(decoder)
    final=Dense(label_count.shape[0],activation='softmax')(decoder1)
    autoencoder = Model(inputs=input_layer, outputs=final)

    autoencoder.compile(metrics=['accuracy'],
                    loss=loss,
                    optimizer='adam')
    history=autoencoder.fit(Xtrain,Ytrain,epochs=ae_epoch,batch_size=32)   
    
    
    Ypred=autoencoder.predict(Xtest)

    conf_mat=confusion_matrix(Ytest.argmax(axis=1),Ypred.argmax(axis=1))
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title("Confusion matrix "+fname+" attacks")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
   
    plt.savefig(fname+"2_autoencoder.png")
