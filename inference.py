import numpy as np

from inference_ner import inference_ner
from inference_cv import inference_cv
from utils.translate import index_to_label
from utils.logger import log


def inference(text, img_path):
    log('Waiting for predictions...')
    # recognise named entities from ner
    doc = inference_ner(text)
    ner_labels = doc.ents

    # get predictions for all classes
    predictions = inference_cv(img_path)
    # get most likely class
    cv_prediction = np.argmax(predictions)
    # convert it to text
    cv_label = index_to_label[cv_prediction]

    return any(cv_label.lower() == ent.text.lower() for ent in ner_labels), doc, predictions
