#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

#pip install opencv-python or pip install opencv-contrib-python
#pip install opencv-python
import cv2

from keras.preprocessing.image import load_img, img_to_array

#conda install -c conda-forge tensorflow 
#conda install -c anaconda keras or conda install -c anaconda keras or nothing
from tensorflow import keras

import tensorflow as tf

from numpy import asarray

from keras.datasets.mnist import load_data
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# # 1. Load the Dataset

# In[5]:


data=load_data() #Returns a tuple
np.shape(data)


# The data has two rows and two columns. 
# 1. First row is the training data and the second row is the testing data
# 2. First column consists of images and the second column is the labels of the images.

# In[6]:


#Unpack the data.
#First row is unpacked into training tuples
#Second row is unpacked into testing tuples
(train_images,train_labels),(test_images,test_labels)=data


# In[7]:


#Print the first training tuple image
train_images[0]


# The above data is represented in the for of 28*28 pixel. There are 28 rows, each comprising of 28 values. The images are stored as numpy arrays

# In[8]:


#Print the first training label
train_labels[0]


# In[9]:


#Put all the labels in a set so as to find all the unqiue labels.
set(train_labels) #or np.unique(train_labels)


# # 2. Visualize the dataset

# To show some images of the dataset with their labels.

# In[10]:


#plt.figure(figsize=(width,height)) is used to set the width and height of the image.  
plt.figure(figsize=(10,10))
for i in range(5):
    #plt.subplot(row,columns,index of the subplot):- used to create a figure.
    plt.subplot(1,5,i+1)#index of the subplot says, which position the subplot will take
    #No xticks or yticks
    plt.xticks([])
    plt.yticks([])
    #plt.imshow() i used to show data as an image.
    plt.imshow(train_images[i],cmap=plt.cm.binary) #cmap used to change the image to binary
    plt.xlabel(train_labels[i]) #Label of the image


# # 3. Data Preprocessing

# Scaling the data. (Convert all the pixel values which are in the range (0-225) to (0-1). This is to reduce the complexity of data and faster training process)

# In[11]:


#Before scaling,the pixels of first training image are as follows:
np.unique(train_images[0])


# In[12]:


train_images=train_images/255.0
test_images=test_images/255.0


# In[13]:


#After scaling, the pixels of first training image are as follows:
np.unique(train_images[0])


# # 4. Build the model

# STEPS TO BUILD THE MODEL:
# 
# 1. Set up the input layer, hidden layer and the output layer
# 2. Complete the model

# Building block of a neural network  in Keras Library is the LAYER. It is a sequential object.
# Each layer consists of several perceptons.

# In[14]:


model=keras.Sequential([
    #Flatten is used to flatten the input layer
    keras.layers.Flatten(input_shape=(28,28)),
    #in Hidden Layer, the first argument is the no. of neurons, chosen at random
    keras.layers.Dense(128,activation='sigmoid'),
    #Last Layer has 10 neurons because there is only 10 output neurons (0-9)
    keras.layers.Dense(10,activation='softmax')
])


# EXAMINE THE STRUCTURE OF WEIGHTS OF THE HIDDEN LAYER

# In[15]:


#To get the hidden layer from the model
hidden_layer=model.layers[1]

#Returns a list of two numpy arrays: 
#1. Matrix of weights.
#2. Array of biases. 
weights=hidden_layer.get_weights()

#Shape of weight: (784,128) signifies that each neuron of input layer is connected to each neuron of the hidden layer
#Shape of biases: 128 signifies the bias of each neuron of hidden layer
print('Shape of weights: ',np.shape(weights[0]))
print('Shape of biases: ',np.shape(weights[1]))


# EXAMINE THE STRUCTURE OF WEIGHTS OF THE OUTPUT LAYER

# In[16]:


#To get the output layer from the model
output_layer=model.layers[2]

#Returns a list of two numpy arrays: 
#1. Matrix of weights.
#2. Array of biases. 
weights=output_layer.get_weights()

#Shape of weight: (128,10) signifies that each neuron of hidden layer is connected to each neuron of the output layer
#Shape of biases: 10 signifies the bias of each neuron of output layer
print('Shape of weights: ',np.shape(weights[0]))
print('Shape of biases: ',np.shape(weights[1]))


# # Compile the model

# Before training, we have to compile the model, otherwise and exception will be thrown during training.
# 1. Loss Functions: 
#     This measures how accurate the model is during training. We will use SPARSE CATEGORICAL CROSS ENTROPY as the loss function
# 2. Optimizer:
#     This is how the model is updated based on the data it sees and its loss function. We will use Stocastic Gradient Descent (SGD)
# 3. Metrics:
#     Used to montior the training and testings steps. The following example uses acuracy, the fraction of images that are correcylt classified.

# In[17]:


#lr= Learning rate
#decay=
#momentum=Technique to drobe sgd faster towards the optimal state. 
sgd=keras.optimizers.SGD(lr=0.5,decay=1e-6,momentum=0.5)
#A metric function is similar to a loss function, except that the results from evaluating a metric are not used when training the model.
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# # 4. Train the model 

# 1. Feed the training data to the model (train_images and train_labels)

# In[18]:


#epochs=no. of timesto itearate over the entire dataset
#batch_size=no. of samples after which the weights are to be updated
#validation_split=ratio of training data to be used as validation data
history=model.fit(train_images,train_labels,epochs=10,batch_size=100,validation_split=0.1)


# # 5. Visualize the training model

# Visulaize validation loss against loss over the training data set per epoch

# The fit() method on a Keras Model returns a History object. The History.history attribute is a dictionary recording training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable)

# In[19]:


#to store validation loss values
val_losses=history.history['val_loss']

#to story training loss values
losses=history.history['loss']

#for x-coordinates of the plot function
indices=range(len(losses))

plt.figure(figsize=(10,5))
plt.plot(indices,val_losses,color='r')
plt.plot(indices,losses,color='g')

#To display Legend
plt.legend(['Validation loss','Loss'])

plt.xlabel('Epochs')
plt.ylabel('Loss')


# If the loss of the training set is less than the loss over the validation set, it is known as Overfitting.
# 
# When the validation loss is slightly higher than the training loss, it's okay. But, if the validation loss is very high than the training loss, it's Overfitting.

# # 6. Compute accuracy and make predictions

# # Evaluate the model by computing the accuracy over testing data

# In[20]:


#Returns the loss value & metrics values for the model in test mode.
test_loss, test_acc=model.evaluate(test_images, test_labels)
print('Test Accuracy', test_acc)
print('Test  Loss', test_loss)


# # Make predictions

# In[ ]:


image = cv2.imread("9w.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(255-gray,(28,28))

blur = cv2.GaussianBlur(gray,(5,5),0)
ret, thresh1 = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

cv2.imwrite("testing.png", thresh1)

data = asarray(thresh1)
data=data/255.0
predictions = model.predict(np.expand_dims(data, 0))


# #Without thresholding or gaussian blurring
# image = cv2.imread("original.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.resize(gray,(28,28))
# 
# cv2.imwrite("testing.png", gray)
# 
# data = asarray(gray)
# data=data/255.0
# predictions = model.predict(np.expand_dims(data, 0))
# #predictions=model.predict(np.expand_dims(test_images[0],0))

# # Define a function to display image along with confidence levels

# Confidence levels show how confident is the model to predict ambiguous images.

# In[365]:


def plot_confidence(images,labels,predictions):
    #15 is the width and 30 is the height of the figure
    plt.figure(figsize=(15,30))
    
    #to set spacing between the plots. hspace=spacing between rows. wspace=spacing between columns.
    plt.subplots_adjust(top=0.99,bottom=0.01,hspace=1.5,wspace=0)
    
    #Location of a particular plot.
    plot_index=1;
    #plt.subplot(no. of rows,no. of columns, plot no.)
    #First columns=images. Second column=bar plot of confidence level
    plt.subplot(1,2,plot_index)
    #Display the image in grayscale
    plt.imshow(images,cmap=plt.cm.binary)
    #Correct label
    correct_label=str(labels)
    #Predicted label is the argument in predictions with highest confidence
    #argmax() Returns the indices of the maximum values along an axis.
    #The value of the prediction will be max on the predicetd label.
    #The predicted label and numpy array's index are same for this problem. 
    predicted_label=str(np.argmax(predictions))
        
    title='Correct label: '+correct_label+'\n'+'Predicted Label: '+predicted_label
    if predicted_label!=correct_label:
        plt.title(title,backgroundcolor='r',color='w')
    else:
        plt.title(title,backgroundcolor='g',color='w')
            
    #To remove the xticks and yticks
    plt.xticks([])
    plt.yticks([])
        
    plt.subplot(1,2,plot_index)
        
    #Display the bar graph with x axis as digits 0-9 and y axis as the predictions of those digits
    plt.bar(range(10),predictions)
    plt.xticks(range(10))
    plt.ylim(0,1) #as the confidence level lies in that range


# In[366]:


images=data
labels=9
test_predictions=predictions
plot_confidence(images,labels,test_predictions[0])


# In[ ]:




