#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.datasets.mnist import load_data


# ## Importing Datasets

# In[ ]:


with np.load(r"C:\Users\Pranav Uniyal\Skifi labs\Datasets\training-images.npz") as data:
    a = data['images']
    train_labels_unshuffled = data['labels']
    
with np.load(r"C:\Users\Pranav Uniyal\Skifi labs\Datasets\testing-images.npz") as data:
    b = data['images']
    test_labels_unshuffled = data['labels']


# ### Unpacking the database

# In[ ]:


#TrainingData
c=[0.2989, 0.5870, 0.1140]
grey_train=np.dot(a,c)
train_images_unshuffled = (255-(np.round(grey_train)).astype(dtype="uint8"))

#TestingData
grey_test=np.dot(b,c)
test_images_unshuffled = (255-(np.round(grey_test)).astype(dtype="uint8"))


# ### Shuffling the Database

# In[ ]:


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# In[ ]:


train_images, train_labels = unison_shuffled_copies(train_images_unshuffled, train_labels_unshuffled)

test_images, test_labels = unison_shuffled_copies(test_images_unshuffled, test_labels_unshuffled)


# ### Plotting an image

# In[ ]:


count=5
plt.figure(figsize=[10,10])
plt.xticks([])
plt.yticks([])
plt.imshow(train_images[count], cmap=plt.cm.binary)
plt.xlabel(train_labels[count])


# ## Data Set Properties

# In[ ]:


print("Max pixel value:",train_images[0].max())
print("Min pixel value:",train_images[0].min())
print()
print("Size of training data:",train_images.shape)
print("Size of an image:",train_images[0].shape)
print()
print("Size of testing data:",test_images.shape)
print("Size of testing data images:",test_images[0].shape)


# ## Visualizing the database
# we will use matplotlib function(imshow) to plot the image.

# In[ ]:


plt.figure(figsize=[10,10])
for i in range(5):
        plt.subplot(3,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])


# # scaling of data
# 
# before scaling

# to scale we need to vector divide the train and test data by 255
# 
# scaling:

# In[ ]:


train_images= train_images  / 255
test_images=test_images / 255


# after scaling:

# In[ ]:


#Neural Network creation
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32)),
    keras.layers.Dense(300, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])


# In[ ]:


#Model Structure
hidden_layer = model.layers[1]
out_weights = hidden_layer.get_weights()

print("Input Layer:")
print("Shape of Weights:", np.shape(out_weights[0]))
print("Shape of Baises:", np.shape(out_weights[1]))
print()

output_layer = model.layers[2]
in_weights = output_layer.get_weights()
print("Output Layer:")
print("Shape of weights:", np.shape(in_weights[0]))
print("Shape of weights:", np.shape(in_weights[1]))


# ## Training the neural network

# In[ ]:


sgd=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history=model.fit(train_images, train_labels, epochs = 5, batch_size =30, validation_split = 0.1)


# ## Visualization of training 

# In[ ]:


val_losses = history.history['val_loss']
losses = history.history['loss']

indices=range(len(losses))

plt.figure(figsize=(10, 5))
plt.plot(indices, val_losses, color='r')
plt.plot(indices, losses, color='g')
plt.legend(['Validation Loss', 'Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')


# ## Compute Accuracy and Evaluate

# In[ ]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print("The test accuracy is:", test_acc)


# In[ ]:


prediction = model.predict(test_images)


# ## Plot the confidence level of predictios

# In[ ]:


def plot_confidence(images, labels, predictions):
    plt.figure(figsize = (15,30))
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)
    p_index = 0
    for i in range(len(images)):
        p_index += 1
        plt.subplot(len(images), 2, p_index)
        plt.imshow(images[i], cmap=plt.cm.binary)
        correct_label = str(labels[i])
        predicted_label = str(np.argmax(predictions[i]))
        title = "Correct Label="+correct_label+"\n"+"Predicted label="+predicted_label
        
        if (correct_label != predicted_label):
            plt.title(title, backgroundcolor='r', color='w')
        else:
            plt.title(title, backgroundcolor='g', color='w')
        
        plt.xticks([])
        plt.yticks([])
        
        p_index += 1
        
        plt.subplot(len(images), 2, p_index)
        plt.bar(range(10), prediction[i])
        plt.xticks(range(10))
        plt.ylim(0,1)


# ## Plotting confidence levels of first 10 images

# In[ ]:


images = test_images[:10]
labels = test_labels[:10]
test_predictions = prediction[:10]
plot_confidence(images, labels, test_predictions)

