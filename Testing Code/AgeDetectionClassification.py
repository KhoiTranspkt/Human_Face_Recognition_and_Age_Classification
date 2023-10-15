import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from keras.preprocessing import image_to_array
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
import os
from pathlib import Path

# Load Cascade Classifier for face detection
filepath_cascade = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\AgeDetection\Codecolab\faceDetectionFront.xml'
face_cascade = cv.CascadeClassifier(filepath_cascade)

# Load the image
filepath_image = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\AgeDetection\Codecolab\70_0_0_20170120223312220.jpg'
img = cv.imread(filepath_image, cv.IMREAD_COLOR)

# Warning 
if img is None:
      print("Error: Could not read the image.")
      exit()

# Convert the image to grayscale
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Detect human faces
faces = face_cascade.detectMultiScale(imgGray,1.3,5)
print(faces)
# Check whether faces were detected or not
if len(faces) == 0:
      print("No faces were detected in the image.")
      exit() 

# Load the pre-trained age prediction model
model_input = tf.keras.Input(shape=(200,200,3))
x = layers.Conv2D(96, 3, strides = (1,1), padding = 'valid', activation = 'relu')(model_input)
x = layers.MaxPool2D(pool_size = (3,3), strides = 2)(x)
x = layers.Conv2D(128, 3, strides = (1,1), padding = 'valid', activation = 'relu')(x)
x = layers.MaxPool2D(pool_size = (3,3), strides = 2)(x) 
x = layers.Conv2D(256, 3, strides = (1,1), padding = 'valid', activation = 'relu')(x)
x = layers.MaxPool2D(pool_size = (3,3), strides = 2)(x)
x = layers.Conv2D(384, 3, strides = (1,1), padding = 'valid', activation = 'relu')(x)
x = layers.MaxPool2D(pool_size = (3,3), strides = 2)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation='relu')(x)
model_output = layers.Dense(23, activation='softmax')(x)

# Load model weights
filepath_model = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\AgeDetection\Model\mode59epochs.h5'
model = tf.keras.Model(inputs=model_input, outputs=model_output )
model.load_weights(filepath_model)

# Define age labels
age_dictionaries = {0:'age1to5', 1:'age6to10', 2:'age11to15', 3:'age16to20', 4:'age21to25', 5:'age26to30', 6:'age31to35', 7:'age36to40', 
        8:'age41to45', 9:'age46to50', 10:'age51to55', 11:'age56to60', 12:'age61to65', 13:'age66to70', 14:'age71to75',
       15:'age76to80', 16:'age81to85', 17:'age86to90', 18:'age91to95', 19:'age96to100', 20:'age101to105', 21:'age106to110', 22:'age111to116'}

# Loop over detected faces
for (x, y, w, h) in faces:
    # Crop and preprocess the face region for age prediction 
        roi_g = imgGray[y:y+h, x:x+w]
        roi_g = cv.resize(roi_g,(48,48), interpolation=cv.INTER_AREA)

        # Normalize to range [0,1]
        roi = roi_g.astype(float)
        cv.normalize(roi,roi,0, 1.0, cv.NORM_MINMAX)

        # Crop roi_c
        roi_c = img[y:y+h, x:x+w]
        roi_c = cv.resize(roi_c,(200,200), interpolation=cv.INTER_AREA)

        # Predict age for each face
        age = model.predict(np.array(roi_c).reshape(-1,200,200,3))
        maxindex = int(np.argmax(age))
        print(age)

        # Draw rectangle around face 
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

        # Dispaly predicted age on the image
        cv.putText(img, 'Class: '+ age_dictionaries[maxindex],(x-10,y-10),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),1)
        
# Save the image with predicted ages
filepath_save = r'E:\PLAN SEMESTER 1 2021-2022\Machine Vision\AgeDetection\result_agedetection'
namefile = '70_AGE_FEMALE.jpg'
cv.imwrite(os.path.join(filepath_save, namefile), img)

# Display the image with predicted ages 
cv.imshow('Age Detection', img)
cv.waitKey()
cv.destroyAllWindows()