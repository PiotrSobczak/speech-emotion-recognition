import pickle
from os.path import isfile

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import label_binarize
import torch

from features import stFeatureExtraction
from helper import pad_sequence_into_array
from train import run_training
from config import AcousticConfig


def calculate_features_3(frames, freq, options):
    # double the window duration
    window_sec = 0.08
    window_n = int(freq * window_sec)

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1

        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f

emotions_used = np.array(['ang', 'exc', 'neu', 'sad'])
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

Y_FILE_NAME = "Y.npy"
X_FILE_NAME = "x_train_speech3.npy"

if not isfile(Y_FILE_NAME) or not isfile(X_FILE_NAME):
    print("Dataset doesn't exist, unpickling dataset...")
    with open("/media/piosobc/Storage Drive/data/iemocap_balanced.pickle", 'rb') as handle:
        data2 = pickle.load(handle)

if not isfile(Y_FILE_NAME):
    print("Y Data doesn't exist, creating data...")
    Y = []
    for ses_mod in data2:
        Y.append(CLASS_TO_ID[ses_mod['emotion']])

    # Y = label_binarize(Y, emotions_used)
    Y = np.array(Y)
    np.save(Y_FILE_NAME, Y)

else:
    print("Y Data exists, loading data...")
    Y = np.load(Y_FILE_NAME)

print("Y Data loaded.")
print(Y.shape)

if not isfile(X_FILE_NAME):
    print("x_train_speech3 Data doesn't exist, creating data...")
    x_train_speech3 = []

    counter = 0
    for ses_mod in data2:
        x_head = ses_mod['signal']
        st_features = calculate_features_3(x_head, framerate, None)
        st_features, _ = pad_sequence_into_array(st_features, maxlen=200)
        x_train_speech3.append(st_features.T)
        counter += 1
        if (counter % 100 == 0):
            print(counter)

    x_train_speech3 = np.array(x_train_speech3)
    np.save(X_FILE_NAME, x_train_speech3)

else:
    print("x_train_speech3 Data exists, loading data...")
    x_train_speech3 = np.load(X_FILE_NAME)

print("x_train_speech3 Data loaded.")
print(x_train_speech3.shape)

cfg = AcousticConfig()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "cuda":
    print("Using GPU. Setting default tensor type to torch.cuda.FloatTensor")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("Using CPU. Setting default tensor type to torch.FloatTensor")
    torch.set_default_tensor_type("torch.FloatTensor")

val_features = x_train_speech3[:1300]
val_labels = Y[:1300]
train_features = x_train_speech3[1300:]
train_labels = Y[1300:]

"""Running training"""
run_training(cfg, train_features, train_labels, val_features, val_labels)

