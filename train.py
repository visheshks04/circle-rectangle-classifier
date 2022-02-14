from cv2 import imshow
import numpy as np
import cv2
import tensorflow as tf

img = cv2.imread('dataset/toy_train/circle/101.jpg') # 64x64 x 3channels
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 64x64
grayscale = grayscale/255.0
print(grayscale)

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

cnn.summary()
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['binary_accuracy', 'binary_crossentropy']
)