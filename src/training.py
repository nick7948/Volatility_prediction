
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from models import build_lstm_model

def train_lstm(data_dir="data/splits", dir="results/models"):

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    os.makedirs(dir, exist_ok=True)
    checkpoint_filepath = os.path.join(dir, "best_model.keras")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1)

    model_run = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint])

    return model, model_run, X_test, y_test
