import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Activation

from utils.data_loader import load_cv_dataset
from utils.paths import CV_MODEL_PATH
from utils.logger import log


def train_cv():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_cv_dataset()

    aug = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0.18,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    log(f'Loading VGG16 base model...')
    vgg16_base = VGG16(
        input_shape=(224, 224, 3),
        include_top=False,
    )

    for layer in vgg16_base.layers:
        layer.trainable = False

    x = vgg16_base.layers[-1].output
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    output = Dense(10, activation='softmax')(x)
    vgg16 = Model(vgg16_base.input, output)

    learning_rate = 0.001
    epochs = 50
    batch_size = 32
    factor = 0.2

    log(f'Compiling CV model...')
    vgg16.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=3, min_lr=1e-7)
    checkpoint_cb = ModelCheckpoint('../task2/vgg16/model_epoch_{epoch:02d}.keras', save_freq='epoch')

    log(f'Training CV model...')
    history = vgg16.fit(
        aug.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint_cb]
    )

    log(f'Saving CV model...')
    vgg16.save(CV_MODEL_PATH)

    return vgg16
