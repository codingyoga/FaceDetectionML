# -*- coding: utf-8 -*-
##################################################################################
##################################################################################
##################################################################################
# Creat dataset of images
import cv2
import numpy as np

# go to github and download raw files of eye and face.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.isOpened()

id=input('enter user id')
sampleNum=0;
while cam.isOpened():
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces: # drawing a rectangle on picutes by for loop.
        sampleNum=sampleNum+1;
        cv2.imwrite("faces_4/"+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # x,y are starting points and x+w,y+h (w=width, h=height) are ending points. 255,0,0 is for blue color.
        cv2.waitKey(100)
    cv2.imshow('Face', img)
    cv2.waitKey(1);
    if(sampleNum>29):
       break
cam.release()
cv2.destroyALLWindows()

##################################################################################
##################################################################################
##################################################################################

# Implementation of face recognition using neural net
#matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import numpy as np
import os
from skimage import io
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# pip install keras
from keras.utils import np_utils
# pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Listing the path of all the images
DatasetPath = []
for i in os.listdir("faces_4"):
    DatasetPath.append(os.path.join("faces_4", i))

# Reading each image and assigning respective labels
imageData = []
imageLabels = []
    
for i in DatasetPath:
   imgRead = io.imread(i,as_grey=True)
   imageData.append(imgRead)
   labelRead = int(os.path.split(i)[1].split(".")[0])
   imageLabels.append(labelRead)
   

# Checking minimum size of imageData's each image
list_val=[]
for i in imageData:
    val=len(i)
    list_val.append(val)
min_size=min(list_val)
min_size

imageDataFin = []
for i in imageData:
   x,y=0,0 
   cropped = i[x: x + min_size, y: y + min_size]
   imageDataFin.append(cropped)
   
c = np.array(imageDataFin)  
c.shape
c[0]

###############################################################################
# Train test model

# Splitting Dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.7, random_state = 1)
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train.shape
X_test.shape 
nb_classes = labelRead+1 # maximum+1 photo id

y_train = np.array(y_train) 
y_test = np.array(y_test)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# Converting each 2d image into 1D vector
X_train = X_train.reshape(len(X_train), min_size*min_size) # 20 numbere of images in train data. and 3 in test data.
X_test = X_test.reshape(len(X_test), min_size*min_size)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# Defining the model
model = Sequential()
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, nb_epoch=500, verbose=1, validation_data=(X_test, Y_test))

# Evaluating the performance
loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
loss
accuracy
predicted_classes = model.predict_classes(X_test)
correct_classified_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_classified_indices = np.nonzero(predicted_classes != y_test)[0]
len(correct_classified_indices)
len(incorrect_classified_indices)


#Save the Model for future use
model.save("Face_CNN_4.hdf5")
#load our model and get a prediction
from keras.models import load_model
model_4 = load_model("Face_CNN_4.hdf5")
model_4.summary()

#########################################################################################
#########################################################################################
# Creat dataset of new/unseen images
import cv2
import numpy as np

# go to github and download raw files of eye and face.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.isOpened()

id=input('enter user id')
sampleNum=0;
while cam.isOpened():
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces: # drawing a rectangle on picutes by for loop.
        sampleNum=sampleNum+1;
        cv2.imwrite("test_4/"+str(sampleNum)+"."+str(id)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # x,y are starting points and x+w,y+h (w=width, h=height) are ending points. 255,0,0 is for blue color.
        cv2.waitKey(100)
    cv2.imshow('Face', img)
    cv2.waitKey(1);
    if(sampleNum>50):
       break
cam.release()
cv2.destroyALLWindows()
#############################################################
#############################################################

# start testing by processing unseen data
#load our model and get a prediction
from keras.models import load_model
model4 = load_model("Face_CNN_4.hdf5")
model4.summary()

#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import numpy as np
import os
from skimage import io
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# pip install keras
from keras.utils import np_utils
# pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

DatasetPath = []
for i in os.listdir("test_4/"):
   DatasetPath.append(os.path.join("test_4/", i))
imageData = []
imageLabels = []

for i in DatasetPath:
    imgRead = io.imread(i,as_grey=True)
    imageData.append(imgRead)
    labelRead = int(os.path.split(i)[1].split(".")[0])
    imageLabels.append(labelRead)
    

min_size=122 # we know this from training dataset.
imageDataFin = []
for i in imageData:
    x,y = 0,0
    cropped = i[x: x + min_size, y: y + min_size]
    imageDataFin.append(cropped) # loop ends here
    
New_Image_c = np.array(imageDataFin)   
New_Image_c.shape  

New_Image_c = New_Image_c.reshape(len(DatasetPath), min_size*min_size)  
New_Image_c = New_Image_c.astype('float32')
New_Image_c /= 255
    
predicted_classes = model4.predict_classes(New_Image_c)
print(predicted_classes)

import pandas as pd
prd=pd.Series(predicted_classes)
prd.value_counts()
prd_t=prd.value_counts(normalize=True)*100
prd_t
max_probability=prd_t.iloc[0:1]
max_probability



