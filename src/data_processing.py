
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_sequences(data, seq_length=5):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i + seq_length, :-1].values
        y = data.iloc[i + seq_length, -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def split_data(data,dir="data/splits", seq_length=5, test_size=0.2):

    os.makedirs(dir, exist_ok=True)
    X, y = create_sequences(data, seq_length=seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    np.save(os.path.join(dir, "X_train.npy"), X_train)
    np.save(os.path.join(dir, "X_test.npy"), X_test)
    np.save(os.path.join(dir, "y_train.npy"), y_train)
    np.save(os.path.join(dir, "y_test.npy"), y_test)
