# -*- coding: utf-8 -*-
"""Rainpre.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17C_4kgxhOjC1a5rHg4o6VOY1NncfDBpv
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

new_df = pd.read_csv('weather.csv').dropna()

new_df['TempDiff'] = (new_df['MaxTemp']-new_df['MinTemp'])
new_df['AvgWindSpeed'] = (new_df['WindSpeed9am']+new_df['WindSpeed3pm'])/2
new_df['AvgHumidity'] = (new_df['Humidity9am']+new_df['Humidity3pm'])/2
new_df['AvgPressure'] = (new_df['Pressure9am']+new_df['Pressure3pm'])/2
new_df['AvgTemp'] = (new_df['Temp9am']+new_df['Temp3pm'])/2
new_df['WindGustSpeed'] = new_df['WindGustSpeed']
new_df['RainToday'] = new_df['RainToday']
new_df['Wind_x_Temp'] = new_df['AvgWindSpeed'] * new_df['AvgTemp']
new_df['Wind_x_Pressure'] = new_df['AvgWindSpeed'] * new_df['AvgPressure']
new_df['Wind_x_Humidity'] = new_df['AvgWindSpeed'] * new_df['AvgHumidity']
new_df['Pressure_x_Humidity'] = new_df['AvgPressure'] * new_df['AvgHumidity']
new_df['Temp_x_Pressure'] = new_df['AvgPressure'] * new_df['AvgTemp']
new_df['Temp_x_Humidity'] = new_df['AvgHumidity'] * new_df['AvgTemp']
new_df['Temp_x_Gust'] = new_df['TempDiff'] * new_df['WindGustSpeed']

Y = new_df['RainToday'].values.tolist()
Y_int = np.zeros(len(Y))
Y = np.array(Y,dtype=str)
Y_int[np.where(Y=='No')[0]] = 0
Y_int[np.where(Y=='Yes')[0]] = 1
Y = to_categorical(Y_int)
Y_int = np.array(Y_int, dtype=int)
X = new_df.drop('RainToday',axis=1).values.tolist()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_int, test_size=0.2)

pc = PCA(n_components=5)

X_train1=X_train
Y_train1=Y_train
X_test1=X_test
X_train = pc.fit_transform(X_train)
X_test = pc.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
sc = logreg.score(X_test,Y_test)
print('Accuracy of Logistic Regression: ', sc)

y_lpred=logreg.predict(X_test)

f1_score(Y_test,y_lpred,average='macro')

f1_score(Y_test,y_lpred,average='micro')

f1_score(Y_test,y_lpred,average='weighted')

recall_score(Y_test,y_lpred,average='macro')

recall_score(Y_test,y_lpred,average='micro')

recall_score(Y_test,y_lpred,average='weighted')

precision_score(Y_test,y_lpred,average='weighted')

precision_score(Y_test,y_lpred,average='micro')

precision_score(Y_test,y_lpred,average='macro')



sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
sc = sgd.score(X_test, Y_test)
print('Accuracy of SGD Classifier: ', sc)

y_sgdpred=sgd.predict(X_test)

f1_score(Y_test,y_sgdpred,average='macro')

f1_score(Y_test,y_sgdpred,average='micro')

f1_score(Y_test,y_sgdpred,average='weighted')

recall_score(Y_test,y_sgdpred,average='macro')

recall_score(Y_test,y_sgdpred,average='micro')

recall_score(Y_test,y_sgdpred,average='weighted')

precision_score(Y_test,y_sgdpred,average='weighted')

KNN = KNeighborsClassifier(n_neighbors=15)
KNN.fit(X_train,Y_train)
sc = KNN.score(X_test, Y_test)
print('Accuracy of KNN Classifier: ', sc)

y_knnpred=KNN.predict(X_test)

f1_score(Y_test,y_knnpred,average='macro')

f1_score(Y_test,y_knnpred,average='micro')

f1_score(Y_test,y_knnpred,average='weighted')

recall_score(Y_test,y_knnpred,average='macro')

recall_score(Y_test,y_knnpred,average='micro')

recall_score(Y_test,y_knnpred,average='weighted')

precision_score(Y_test,y_sgdpred,average='weighted')





parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['rbf'], 'C': [1, 10, 100, 1000]}]
print("hyper-parameters")


clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, Y_train)
y_svmpred=clf.predict(X_test)
f1_score(Y_test,y_svmpred,average='macro')
f1_score(Y_test,y_svmpred,average='micro')
f1_score(Y_test,y_svmpred,average='weighted')
recall_score(Y_test,y_svmpred,average='macro')
recall_score(Y_test,y_svmpred,average='micro')
recall_score(Y_test,y_svmpred,average='weighted')





random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
sc = random_forest.score(X_test, Y_test)
print('Accuracy of Random Forest Classifier: ', sc)

#importances=pd.DataFrame(({'feature':Xtraindf.columns,'importance':np.round(random_forest.feature_importances_,3)}))

#importances = importances.sort_values('importance',ascending=False).set_index('feature')
#importances

#importances.plot.bar()



y_rfpred=random_forest.predict(X_test)
f1_score(Y_test,y_rfpred,average='macro')

f1_score(Y_test,y_rfpred,average='micro')

f1_score(Y_test,y_rfpred,average='weighted')

recall_score(Y_test,y_rfpred,average='macro')

recall_score(Y_test,y_rfpred,average='micro')

recall_score(Y_test,y_rfpred,average='weighted')

precision_score(Y_test,y_rfpred,average='weighted')

import tensorflow.keras as keras
import tensorflow as tf

import io
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
neuraldf = pd.read_csv('weather.csv').dropna()

neuraldf
neuraldf.loc[neuraldf['RainToday'] =='Yes','RainToday']=1

neuraldf.loc[neuraldf['RainToday'] =='No','RainToday']=0

neuraldf

traindat,testdat=train_test_split(neuraldf,test_size=0.2)

traindat.columns.name='id'
traindat

traindat.shape

testdat.columns.name='id'
testdat

testdat.shape

y_train_neu=traindat['RainToday']
y_train_neu

y_train_neu.shape

y_train_neu=y_train_neu.reshape(-1,1)

y_train_neu.shape

X_train_neu=traindat.drop(['RainToday'],axis=1)

X_train_neu

X_train_neu.shape

Y_true_neu=testdat['RainToday']

Y_true_neu

Y_true_neu=Y_true_neu.reshape(-1,1)
Y_true_neu.shape

X_test_neu=testdat.drop(['RainToday'],axis=1)
X_test_neu

X_test_neu.shape

X_train_neu.shape

y_train_neu.shape

X_test_neu.shape

Y_true_neu.shape
from keras.utils import np_utils

Y_TRAIN_NEU=np_utils.to_categorical(y_train_neu, 2)
Y_TRUE_NEU=np_utils.to_categorical(Y_true_neu, 2)

Y_TRAIN_NEU

Y_TRUE_NEU

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, input_dim=12, kernel_initializer='normal', activation='relu'))
model.add(tf.keras.layers.Dense(8,activation='relu'))
model.add(tf.keras.layers.Dense(4,activation='relu'))
model.add(tf.keras.layers.Dense(2,kernel_initializer='normal',activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) 
model.fit(X_train_neu, Y_TRAIN_NEU, epochs=2000,verbose=1,batch_size=64)

y_nnpred=model.predict(X_test_neu)

y_nnpred

pred = []
for i in range(24077):
  pred.append(np.argmax(y_nnpred[i]))

pred

Y_true_neu.shape

pred=np.array(pred,dtype=int)
pred.shape

pred.reshape(-1,1)
pred.shape

pred

f1_score(Y_true_neu,pred,average='macro')

f1_score(Y_true_neu,pred,average='micro')

f1_score(Y_true_neu,pred,average='weighted')

recall_score(Y_true_neu,pred,average='macro')

recall_score(Y_true_neu,pred,average='micro')

recall_score(Y_true_neu,pred,average='weighted')

precision_score(Y_true_neu,pred,average='weighted')

