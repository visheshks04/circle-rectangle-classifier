import tensorflow as tf
import sys
import os
import cv2

if len(sys.argv) == 1:
    path = 'IMAGE PATH HERE'
else:
    path = sys.argv[1]

model = tf.keras.models.load_model('cls')

predictions = []
for img in os.listdir(path):
    