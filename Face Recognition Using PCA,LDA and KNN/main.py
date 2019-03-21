from FisherFaces import Fisherfaces
from KNNclassifier import NearestNeighbour
from ModelGeneration import PredictableModel,save_model,load_model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#path = 'attfaces/'
path = 'CroppedYale/'
foldernames = []
X = []
Y = []
c = 1
for dirname, dirnames, filenames in os.walk(path):
    for subdirname in dirnames:
        foldernames.append(subdirname)
        subject_path = os.path.join(dirname,subdirname)
        for filename in os.listdir(subject_path):
            try:
                im = cv2.imread(os.path.join(subject_path,filename), cv2.IMREAD_GRAYSCALE)
                #print(filename)
                im = cv2.resize(im, (84,96))
                X.append(np.asarray(im, dtype=np.uint8))
                Y.append(c)
            except:
                #print('Unexpected Error')
                pass

        c = c+1

print('Completed scanning Images')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

print('Completed Test Train split')

feature = Fisherfaces()
classifier = NearestNeighbour(k=1)

my_model = PredictableModel(feature=feature, classifier=classifier)
my_model.compute(X_train, Y_train)

print('Completed training model')

predictions = []
correct_prections = 0
for item in X_test:
    predictions.append(my_model.predict(item))
#predictions = my_model.predict(Test)
for i in range(len(X_test)):
    if Y_test[i] == predictions[i][0]:
        correct_prections += 1
    #print(Y_test[i],predictions[i][0])

print('accuracy: ',(correct_prections/float(len(X_test))))
