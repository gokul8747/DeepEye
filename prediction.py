from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np

class Prediction():
    def __init__(self):
        self.model_path = r'D:\DeepEye\Model\mobilenet.h5'

    def classify(self,img):
        img = img / 255.0
        img = np.expand_dims(img,0)
        model = load_model(self.model_path,compile=False)
        pred = model.predict(img)
        result = pred.argmax()

        return result