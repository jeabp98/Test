# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 07:54:22 2019

@author: A6328
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn import metrics
import seaborn as sns

def readFile(file):
    head=["Name","mcg","gvh","alm","mit","erl","pox","vac","nuc","rpta"] 
    f=pd.read_csv(file,sep=r"\s+")
    f.columns=head
    f.drop(['Name'],1,inplace=True)
    return f

def plot_corr(datos):
    correlation=datos.corr()
    #print(correclation)
    plt.figure(figsize=(15,10))
    sns.heatmap(correlation,annot=True,cmap='coolwarm')
    
def plotcm(conf,labels):
    plt.figure(figsize=(9,9))
    sns.heatmap(conf,annot=True,fmt=".3f",linewidth=5,square=True,cmap='Blues_r',xticklabels=labels,yticklabels=labels)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title("CM",size=15)

def plot_roc(clf):
    plt.plot(clf.loss_curve_)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

def main():
    train="yeast.data"
    x=readFile(train)
    y=x["rpta"]
    x=x.iloc[:,:-1]
    print(x)
    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
    clf=LogisticRegression(max_iter=3000,multi_class="multinomial",solver="sag")
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    conf=confusion_matrix(y_test,y_pred)
    print(conf)

    score=clf.score(x_test,y_test)
    print(score)
    plot_corr(x)
    plotcm(conf,x.columns)



    

if __name__=="__main__":
    main()