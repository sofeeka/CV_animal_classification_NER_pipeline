import random

import spacy
from spacy.training.example import Example

from utils.data_loader import load_ner_dataset
from utils.paths import NER_DATA_PATH, NER_MODEL_PATH
from utils.logger import log


def train_ner():
    train_data = load_ner_dataset(NER_DATA_PATH)

    # load spacy base
    log(f'Loading spacy en_core_web_sm base model...')
    nlp = spacy.load("en_core_web_sm")

    # add a new NER pipeline if it doesn't exist
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # add a new label
    ner.add_label("ANIMAL")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # disable not needed pipes
    log('Training NER model...')
    with nlp.disable_pipes(*other_pipes):
        # train for 10 iterations
        for epoch in range(10):
            # shuffle the data randomly
            random.shuffle(train_data)
            losses = {}

            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # train model
                nlp.update([example], drop=0.5, losses=losses)

            log(f'Epoch {epoch + 1}, Loss: {losses}')

    log(f'Saving the model to {NER_MODEL_PATH}...')
    nlp.to_disk(NER_MODEL_PATH)

    return nlp
