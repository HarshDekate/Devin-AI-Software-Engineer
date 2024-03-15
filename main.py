

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#initializing cnn
classifier = Sequential()

#step 1 - convolution and polling
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))

#adding Dropout
classifier.add(Dropout(0.5))

#ADDING 2ND CONVOLUTION and polling
classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))

#step3 Flatten
classifier.add(Flatten())

#creating ANN
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dense(units=8,activation='softmax'))

#complie the CNN
classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images1
from keras.preprocessing.image import ImageDataGenerator

#radom scaling are applied to the images before training the model
#image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
samplewise_center=True,
vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64,64),#it should same as the input shape in convolaion
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/train',
        target_size=(64,64),#it should same as the input shape in convolaion
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

classifier.fit_generator(training_set,
        steps_per_epoch=7999,#no of images in training set
        epochs=1,
        validation_data=test_set,
        validation_steps=4000)

import numpy as np
import os
from keras.preprocessing import image

# specify the path to your 'data/test' directory
test_dir = 'data/test'

# get the list of all directories in your 'data/test' directory
folders = os.listdir(test_dir)

for folder in folders:
    # get the list of all images in the current directory
    images = os.listdir(os.path.join(test_dir, folder))
    
    for img in images:
        # construct the full image path
        img_path = os.path.join(test_dir, folder, img)
        
        # load the image
        test_img = image.load_img(img_path, target_size=(64,64), color_mode='grayscale')
        
        # process the image (if necessary)
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        
        # use the image for prediction
        result = classifier.predict(test_img)
        
        # print or save the result (if necessary)
        print(result)


#saving the model using it along with opencv 
from keras.models import load_model
classifier.save('hand_gestures_1.h5') #name of the model








