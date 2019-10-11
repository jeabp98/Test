# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:05:48 2019

@author: a6327
"""
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn import preprocessing #Normalizacion datos vacios
from sklearn.neural_network import MLPClassifier # Para clasificacion entre un conjunto de clases
from sklearn.model_selection import train_test_split #Division entre el conjunto de data de entrenamiento
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import  cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns




def abrirArchivo(fileR):
    head=["Name","mcg","gvh","alm","mit","erl","pox","vac","nuc","rpta"] 
    f=pd.read_csv(fileR,sep=r"\s+")
    f.columns=head
    f.drop(['Name'],1,inplace=True)
    nn(f)
    
def nn(f): #Funcion para la red neuronal
    X=f.iloc[:,:-1]
    y=f["rpta"]

    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3) #Dividir los datos de conjunto x de prueba y "y" de prueba
    #Lo de aca abajo es el modelo de red neuronal
    mlp=MLPClassifier(activation='relu', batch_size=4,hidden_layer_sizes=20,learning_rate="constant",max_iter=3000,solver="adam") #Utilizar una red neuronal max_iter es la cantidad de epocas
    
    print ("============train================")
    print (mlp.fit(X_train,y_train))  #Entrena con el modelo y aprendete la respuesta
    print (mlp.score(X_train,y_train))
    
    predictions=mlp.predict(X_train)
    print(confusion_matrix(y_train,predictions))
    
    plt.plot(mlp.loss_curve_)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    
    print ("============test================")
    predictions=mlp.predict(X_test)
    conf=confusion_matrix(y_test,predictions)
    print(conf)
    print (mlp.score(X_test,y_test))
    plot_corr(X)
    plotcm(conf,[1,2,3,4,5,6,7,8,9])


def onehot(f,indices):
    f=pd.concat([f,pd.get_dummies(f[indices])], axis=1)
    f=f.drop(indices,1)
    return f

def plotcm(conf,labels):
    plt.figure(figsize=(9,9))
    sns.heatmap(conf,annot=True,fmt=".3f",linewidth=5,square=True,cmap='Blues_r',xticklabels=labels,yticklabels=labels)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title("CM",size=15)

def plot_corr(datos):
    correlation=datos.corr()
    #print(correclation)
    plt.figure(figsize=(15,10))
    sns.heatmap(correlation,annot=True,cmap='coolwarm')

def main():
    f="yeast.data"
    abrirArchivo(f)

if __name__=="__main__":
    main()