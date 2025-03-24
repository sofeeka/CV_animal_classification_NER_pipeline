import cv2
import numpy as np
from tensorflow import keras

from utils.paths import CV_MODEL_PATH
from utils.logger import log


def preprocess_image(img, size=(224, 224)):
    # convert from bgr to rgb because cv2 reads image in bgr
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize to suitable size
    image = cv2.resize(image, size)
    # normalize pixels
    image = image / 255.0
    # loaded image has shape (224, 224, 3), we need (1, 224, 224, 3) where 1 is the batch size
    image = np.expand_dims(image, axis=0)
    # make sure its float
    image = image.astype(np.float32)
    return image


def inference_cv(img_path, size=(224, 224)):
    log(f'Reading image {img_path}')
    img = cv2.imread(img_path)
    log(f'Preprocessing image...')
    img = preprocess_image(img, size=size)

    log(f'Loading CV model...')
    vgg16 = keras.models.load_model(CV_MODEL_PATH)
    log(f'Making predictions...')
    predictions = vgg16.predict(img)
    return predictions
