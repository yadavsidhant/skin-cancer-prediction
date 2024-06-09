import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from google.colab import drive
import cv2
import json

# Mount Google Drive
drive.mount('/content/drive')

# Unzip dataset
!unzip -q '/content/drive/MyDrive/HAM10000/HAM10000_images_part_1.zip' -d ./HAM10000_images
!unzip -q '/content/drive/MyDrive/HAM10000/HAM10000_images_part_2.zip' -d ./HAM10000_images
!unzip -q '/content/drive/MyDrive/HAM10000/ISIC2018_Task3_Test_Images.zip' -d ./ISIC2018_Test_Images

# Load metadata
metadata = pd.read_csv('/content/drive/MyDrive/HAM10000/HAM10000_metadata.csv')
test_metadata = pd.read_csv('/content/drive/MyDrive/HAM10000/ISIC2018_Task3_Test_GroundTruth.csv')

image_dir = './HAM10000_images'
metadata['image_path'] = metadata['image_id'].apply(lambda x: os.path.join(image_dir, f'{x}.jpg'))

test_image_dir = './ISIC2018_Test_Images'
test_metadata['image_path'] = test_metadata['image_id'].apply(lambda x: os.path.join(test_image_dir, f'{x}.jpg'))

print(metadata.head())
print(test_metadata.head())

# Data Augmentation
IMAGE_SIZE = 128
BATCH_SIZE = 64

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20
)

train_generator = datagen.flow_from_dataframe(
    dataframe=metadata,
    x_col='image_path',
    y_col='dx',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=metadata,
    x_col='image_path',
    y_col='dx',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_metadata,
    x_col='image_path',
    y_col='dx',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Model Building
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=14
)

# Save model and class indices
os.makedirs('./model', exist_ok=True)
model.save('./model/skin_cancer_model.h5')

class_indices = train_generator.class_indices
with open('./model/class_indices.json', 'w') as json_file:
    json.dump(class_indices, json_file)
