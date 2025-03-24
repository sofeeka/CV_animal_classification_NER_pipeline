# Task 2. Named entity recognition + image classification

---

## Meeting the requirements

In this task I have created a system that consists of 2 models: NER and Image Recognition.

`spacy` and `en_core_web_sm` base model were used to solve NLP part of the task.

`tensorflow` and `VGG16` base model were the main components of solving CV part of the task.

### 1. Finding and creating data to use for training

##### CV and Image Classification with VGG-16 base

I have found `Animal-10` dataset. It was used to train Image Classification model.

##### NLP and Named Entity Recognition

Using LLM I have generated 20 sentences for every animal present in Animals-10 dataset. These marked sentences were used
to train NER model.

---

##### Animals present in the solution:

- dog
- cat
- cow
- horse
- sheep
- spider
- chicken
- elephant
- squirrel
- butterfly

---

### 2. Solution structure

**train and inference.py** files for NER model

- `train_ner.py` and `inference_ner.py`

**train and inference.py** files for Image Classification model

- `train_cv.py` and `inference_cv.py`

**the entire pipeline** that takes 2 inputs

- `inference.py` with `inference(text, img_path)` method

**additionally:**

- `demo.ipynb` showcasing the work of the pipeline
- `dataset_exploratory.ipynb` where data collecting / preprocessing is showcased and explained
- `utils/logger.py` with `log()` method and `logging` boolean. `logging=True` by default.

---

## Installation Guide

### 1. Prerequisites

Ensure you have the following installed:

- Python (3.10-3.12)
- Git
- Virtual environment tools (venv or conda)

### 2. Clone the repository

```bash
git clone <https://github.com/sofeeka/CV_animal_classification_NER_pipeline>
cd <CV_animal_classification_NER_pipeline>
```

### 3. Set Up Virtual Environment

##### Linux

```bash
python -m venv venv
source venv/bin/activate
```

##### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Get CV Image Classification Model

1. Download it from Google Drive
   [link to model file](https://drive.google.com/file/d/1kmM3VJGTu5a3vYOHmxJ9Xu9rCYsdP9dl/view?usp=sharing)
2. Update `CV_MODEL_PATH` in `utils/paths.py` with your model
   address. `CV_MODEL_PATH='vgg16/vgg16_model.keras'` by default
