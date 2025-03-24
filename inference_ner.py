import os
import spacy

from train_ner import train_ner
from utils.paths import NER_MODEL_PATH
from utils.logger import log


def inference_ner(text):
    log(f'Attempting to load NER model...')
    # check if model exists
    if not os.path.exists(NER_MODEL_PATH):
        # train new model
        log(f'Model not found. Training new model...')
        nlp = train_ner()
    else:
        # load existing model
        log(f'Loading existing NER model...')
        nlp = spacy.load(NER_MODEL_PATH)

    log(f'Extracting named entities...')
    doc = nlp(text)
    return doc


def pretty_print_doc(doc):
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    if not doc.ents:
        print('No entities found')
