import os
import cv2
import json
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.paths import NER_DATA_PATH, CV_DATA_PATH
from utils.translate import label, label_to_index
from utils.logger import log


def load_ner_dataset(file_path=NER_DATA_PATH):
    log('Loading NER dataset...')
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def preprocess_image(image_path, size=(224, 224)):
    # read image
    image = cv2.imread(image_path)
    # cv2 reads image in BGR so convert it to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize to make sure dimensions are the same for all the pictures
    image = cv2.resize(image, size)
    return image


def load_cv_dataset(file_path=CV_DATA_PATH):
    log('Loading Animals-10 dataset for CV...')
    # download a dataset if it is not present
    if os.path.exists(file_path):
        root = file_path
    else:
        root = kagglehub.dataset_download("alessiocorrado99/animals10")
    root = os.path.join(root, 'raw-img')

    img_paths = []
    labels = []

    # for each animal (separate folder for each)
    for animal in os.listdir(root):
        # join the animal folder to path
        path = os.path.join(root, animal)
        # add first 1000 images for each animal to balance the dataset
        for img in os.listdir(path)[:1000]:
            # add image path
            img_paths.append(os.path.join(path, img))
            # add label (animal name)
            labels.append(animal)

    df = pd.DataFrame({
        'img_path': img_paths,
        'label': labels
    })

    # add label indices
    df['label_num'] = df['label'].map(label_to_index)
    df['label'] = df['label'].map(label)

    # preprocess images
    all_images = []
    for i in df['img_path'].values:
        image = preprocess_image(i)
        all_images.append(image)

    all_images = np.array(all_images)

    # all data split to 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, df['label_num'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_num']
    )

    # temp split to 50% validation, 50% test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test
