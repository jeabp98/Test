from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn import preprocessing #Normalizacion datos vacios
from sklearn.neural_network import MLPClassifier # Para clasificacion entre un conjunto de clases
from sklearn.model_selection import train_test_split, GridSearchCV #Division entre el conjunto de data de entrenamiento
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import  cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def readFile(file):
    head=["Name","mcg","gvh","alm","mit","erl","pox","vac","nuc","rpta"] 
    f=pd.read_csv(file,sep=r"\s+")
    f.columns=head
    f.drop(['Name'],1,inplace=True)
    return f
    

def test(X,y):
    #Training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    #Dummy Classifier
    clf = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print('y actual : \n' +  str(y_test.value_counts()))

    #Distribution of y predicted
    print('y predicted : \n' + str(pd.Series(y_pred).value_counts()))
    print(str(accuracy_score(y_test,y_pred)))
    print('Confusion Matrix: \n' + str(confusion_matrix(y_test,y_pred)))
    
    #Logistic Regression
    clf = LogisticRegression(max_iter=3000,multi_class="multinomial",solver="sag").fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    #Evaluation Metrics
    print('Accuracy Score: \n' + str(accuracy_score(y_test,y_pred)))
    
    #Logistic Regerssion Confusion Matrix
    print('Confusion Matrix \n' + str(confusion_matrix(y_test,y_pred)))

    #Grid Search
    clf = LogisticRegression(max_iter=3000,multi_class="multinomial",solver="sag")
    grid_values = {'penalty': ['none'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall',average='micro')
    grid_clf_acc.fit(X_train, y_train)
#    
#    #Predict values based on new parameters
#    y_pred_acc = grid_clf_acc.predict(X_test)
#    
#    # New Model Evaluation metrics 
#    print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
#
#    #Logistic Regression (Grid Search) Confusion matrix
#    print(confusion_matrix(y_test,y_pred_acc))

def main():
    file = "yeast.data"
    X= readFile(file)
#    print(X['rpta'].value_counts())
    y=X["rpta"]
#    print(y)
    X=X.iloc[:,:-1]
    test(X,y)
    
    
if __name__ == "__main__":
    main()