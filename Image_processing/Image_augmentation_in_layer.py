# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 23:23:49 2021

@author: ensl9
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# =============================================================================
# https://www.tensorflow.org/tutorials/images/data_augmentation?hl=ko
# https://lynnshin.tistory.com/27
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing?hl=ko
# https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/#zooming
# =============================================================================

#%% dataset setting
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
# _ = plt.imshow(image)
# _ = plt.title(get_label_name(label))


#%% image resize
IMG_SIZE = 180
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])
# result = resize_and_rescale(image)
# _ = plt.imshow(result)
# 배율 조정 레이어는 픽셀 값을 [0,1]로 표준화합니다. 
#[-1,1]로 표준화를 원할 경우, Rescaling(1./127.5, offset=-1)


#%% augmentation
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomContrast((0.5,0.5)),
  layers.experimental.preprocessing.RandomZoom(0.3)
])

# # augmented image visualization
# image = tf.expand_dims(image, 0)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#   augmented_image = data_augmentation(image)
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(augmented_image[0])
#   plt.axis("off")


#%% plug augmentation into model definition
model = tf.keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model
])
