#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import keras 
from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten 
from keras import backend as k 

import cv2
from numpy import asarray
import matplotlib.pyplot as plt


# In[12]:


(x_train, y_train), (x_test, y_test) = mnist.load_data() 


# In[13]:


img_rows, img_cols=28, 28

if k.image_data_format() == 'channels_first': 
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
    inpx = (1, img_rows, img_cols) 

else: 
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
    inpx = (img_rows, img_cols, 1) 

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255


# In[14]:


y_train = keras.utils.to_categorical(y_train) 
y_test = keras.utils.to_categorical(y_test) 


# In[15]:


inpx = Input(shape=inpx) 
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx) 
layer2 = Conv2D(64, (3, 3), activation='relu')(layer1) 
layer3 = MaxPooling2D(pool_size=(3, 3))(layer2) 
layer4 = Dropout(0.5)(layer3) 
layer5 = Flatten()(layer4) 
layer6 = Dense(250, activation='sigmoid')(layer5) 
layer7 = Dense(10, activation='sigmoid')(layer6) 


# In[16]:


model = Model([inpx], layer7) 
model.compile(optimizer=keras.optimizers.Adadelta(), 
			loss=keras.losses.categorical_crossentropy, 
			metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=12, batch_size=500) 


# In[86]:


score = model.evaluate(x_test, y_test, verbose=0) 
print('loss=', score[0]) 
print('accuracy=', score[1]) 


# image = cv2.imread("twow.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.resize(255-gray,(28,28))
# 
# blur = cv2.GaussianBlur(gray,(5,5),0)
# ret, thresh1 = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
# 
# cv2.imwrite("testing.png", thresh1)
# 
# data = asarray(thresh1)
# data=data/255.0
# data = data.reshape(1, 28, 28, 1)
# predictions = model.predict((data))

# In[100]:


#Without thresholding or gaussian blurring
image = cv2.imread("ninew.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(255-gray,(28,28))

cv2.imwrite("testing.png", gray)

data = asarray(gray)
data=data/255.0
data = data.reshape(1, 28, 28, 1)
predictions = model.predict(data)


# In[101]:


images=data
labels=2
test_predictions=predictions
print(np.argmax(predictions))
#plot_confidence(images,labels,test_predictions[0])


# 
