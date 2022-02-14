import tensorflow as tf
import sys
import os
import cv2
from data import preproc

def classifier(path):

    model = tf.keras.models.load_model('cls')

    predictions = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = preproc(img)
        img = tf.convert_to_tensor([img])
        tf.reshape(img, (64,64,1))
        pred = int(model.predict(img).item() > 0.5)
        predictions.append(pred)

    return predictions


if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        path = 'PATH TO IMAGES DIRECTORY HERE'
    else:
        path = sys.argv[1]

    predictions = classifier(path)

    print(predictions)