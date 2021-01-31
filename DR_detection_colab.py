# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:23:46 2021

@author: Ratul
"""

from keras.layers import Input,Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
IMAGE_SIZE=[224,224]
train_path='/content/drive/MyDrive/TRAIN'
test_path='/content/drive/MyDrive/TEST'



vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
for layers in vgg.layers:
  layers.trainable=False


x=Flatten()(vgg.output)
prediction=Dense(5,activation='softmax')(x)
model=Model(inputs=vgg.input,outputs=prediction)
model.summary()


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('/content/drive/MyDrive/TRAIN',target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=test_datagen.flow_from_directory('/content/drive/MyDrive/TEST',target_size=(224,224),batch_size=32,class_mode='categorical')


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(training_set,steps_per_epoch=16, validation_data=test_set, validation_steps=8,epochs=5)





import os
import cv2
import numpy as np
lst=os.listdir("/content/drive/MyDrive/TEST/TEST_4")
src='/content/drive/MyDrive/TEST/TEST_4/'
xxxx=[]
for i in lst:
    img=cv2.imread(src+i)
    img_t = cv2.resize(img, (224, 224)) 
    img_tt=img_t/255.0
    img_ttt=np.expand_dims(img_tt,axis=0)
    xx=model.predict_generator(img_ttt)
    xxx=np.argmax(xx)
    xxxx.append(xxx)


from collections import Counter
cnt=Counter(xxxx).keys() # equals to list(set(words))
vals=Counter(xxxx).values()
print(cnt,vals)
