#TUTORIAL
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
#IMPORTAR y VER COMO ESTA COMPUESTO EL DF
df=pd.read_csv('C:/Users/jsala/OneDrive - Universidad Adolfo Ibanez/QUANTUM BLACK/Eopinions.csv')
pd.set_option('display.max_colwidth',None)
print(df['text'].str.count(' '))
print(df['text'].str.split().str.len())
print(df['text'].str.split().str.len().mean())
print(df['text'].str.split().str.len().describe())






print('Conteo por categoría')
print((df['class'].value_counts()))
print('HAY ALGUN VALOR NULL? ')
print(df.isnull().values.any(),'\n')

print('suma de valores nulos: ')
print(df.isnull().sum(),'\n')

#SEPARA DATASET DE ENTRENAMIENTO Y PRUEBA
train,test=train_test_split(df,test_size=0.33,random_state=42)

print(len(train))
print(len(test),'\n')

print(train['class'].value_counts(),'\n')
print(test['class'].value_counts())

#DEFINIR VARIABLES
x_train=train['text'].to_list()
x_test=test['text'].to_list()
y_train=train['class'].to_list()
y_test=test['class'].to_list()

#VECTORIZAR
vectorizer=CountVectorizer()
x_trainvec=vectorizer.fit_transform(x_train)
x_testvec=vectorizer.transform(x_test)
print('shape de vector de entrenemiento')
print(x_trainvec.shape,'\n')

print('shape de vector de prueba')
print(x_testvec.shape,'\n')

#DEFINIR MODELO Y ENTRENAMIENTO
from sklearn.tree import DecisionTreeClassifier 
clf_dec=DecisionTreeClassifier()
clf_dec.fit(x_trainvec,y_train)

print(x_test[3],'\n')

#PREDECIR
print(clf_dec.predict(x_testvec[3]),'\n')


#METRICAS DEL MODELO
print(clf_dec.score(x_testvec,y_test))

from sklearn.metrics import f1_score
puntaje=f1_score(y_test,clf_dec.predict(x_testvec),average=None)
print('PUNTAJE F1: ')
print(puntaje)
from sklearn.metrics import confusion_matrix
matriz=confusion_matrix(y_test,clf_dec.predict(x_testvec))
print('MATRIZ DE CONFUSION: ')
print(matriz)

test_prueba=['I like this for travel to my hometown','it take very good images, im goingo to buy a new filter','It is very usefull to drive in the city']
new_test=vectorizer.transform(test_prueba)
a=(clf_dec.predict(new_test))
print(a)

#MODELO SVM - DEFINIR-ENTRENAR-PREDECIR-METRICA
from sklearn import svm
clf_svm= svm.SVC(kernel='linear')
clf_svm.fit(x_trainvec,y_train)
print(clf_svm.predict(x_testvec[0]))
print(clf_svm.score(x_testvec,y_test))
