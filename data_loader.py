import pickle
import json
import numpy as np
from os.path import isfile
from python_speech_features import mfcc

from word2vec_wrapper import Word2VecWrapper
from preprocessing import Preprocessor
from utils import timeit


IEMOCAP_PATH = "data/iemocap.pickle"
IEMOCAP_BALANCED_PATH = "data/iemocap_balanced.pickle"
TRANSCRIPTIONS_PATH = "data/iemocap_transcriptions.json"
TRANSCRIPTIONS_VAL_PATH = "data/iemocap_transcriptions_val.json"
TRANSCRIPTIONS_TRAIN_PATH = "data/iemocap_transcriptions_train.json"
MFCC_FEATURES_PATH = "data/mfcc_features.npy"
MFCC_LABELS_PATH = "data/mfcc_labels.npy"
VAL_SIZE = 1531

USED_CLASSES = ["neu", "hap", "sad", "ang", "exc"]
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}


def save_iemocap_transcriptions():
    """Load IEMOCAP and save transcriptions only in format: [{"emotion":E, "transcription": t},..] """
    iemocap = pickle.load(open(IEMOCAP_BALANCED_PATH, "rb"))
    transcriptions = [{"transcription": dic["transcription"], "emotion": dic["emotion"]} for dic in iemocap]
    with open(TRANSCRIPTIONS_PATH, "w") as file:
        json.dump(transcriptions, file)
    with open(TRANSCRIPTIONS_VAL_PATH, "w") as file:
        json.dump(transcriptions[:VAL_SIZE], file)
    with open(TRANSCRIPTIONS_TRAIN_PATH, "w") as file:
        json.dump(transcriptions[VAL_SIZE:], file)

@timeit
def save_iemocap_mfcc_features(num_mfcc_features=32, window_len=0.025, winstep=0.01, sample_rate=16000, seq_len=800):
    """
    https://arxiv.org/pdf/1706.00612.pdf
    The mean length of all turns is 4.46s (max.: 34.1s, min.: 0.6s). S
    Since the input length for a CNN has to be equal for all samples,
    we set the maximal length to 7.5s (mean duration plus standard deviation).
    Longer turns are cut at 7.5s and shorter ones are padded with zeros.

    speech_durations = [obj['end'] - obj[0]['start'] for obj in iemocap]
    max_duration = np.argmax(np.array(speech_durations)) # 34.1388s
    min_duration = np.argmin(np.array(speech_durations)) # 0.5849s
    avg_duration = sum(seq_lens)/len(seq_lens)           # 4.549 s
    std_dev = np.array(seq_lens).std()                   # 3.23 s
    chosen_duration = ~ avg_duration + std_dev = 8s
    """
    iemocap = pickle.load(open(IEMOCAP_BALANCED_PATH, "rb"))

    labels = np.zeros(len(iemocap))
    mfcc_features_dataset = np.zeros((len(iemocap), seq_len, num_mfcc_features))

    for i, obj in enumerate(iemocap):
        class_id = CLASS_TO_ID[obj["emotion"]]
        labels[i] = class_id
        mfcc_features = mfcc(obj["signal"], sample_rate, window_len, winstep)
        mfcc_features = mfcc_features[:seq_len]
        mfcc_features_dataset[i, :mfcc_features.shape[0], :mfcc_features.shape[1]] = mfcc_features

    np.save(MFCC_FEATURES_PATH, mfcc_features_dataset)
    np.save(MFCC_LABELS_PATH, labels)

@timeit
def load_mfcc_dataset():
    if not isfile(MFCC_FEATURES_PATH) or not isfile(MFCC_LABELS_PATH):
        print("Dataset not found. Creating dataset...")
        save_iemocap_mfcc_features()
        print("Dataset created. Loading dataset...")

    mfcc_features = np.load(MFCC_FEATURES_PATH)
    mfcc_labels = np.load(MFCC_LABELS_PATH)
    print("Dataset loaded.")

    assert mfcc_features.shape[0] == mfcc_labels.shape[0]

    return mfcc_features, mfcc_labels


def create_balanced_iemocap():
    """Keeping only Neutral, Happiness, Sadness, Anger classes. Merging excited samples into happiness class"""
    iemocap = pickle.load(open(IEMOCAP_PATH, "rb"))
    balanced_iemocap = []
    for dic in iemocap:
        if dic["emotion"] in USED_CLASSES:
            if dic["emotion"] == "exc":
                dic["emotion"] = "hap"
            balanced_iemocap.append(dic)
    with open(IEMOCAP_BALANCED_PATH, "wb") as file:
        pickle.dump(np.array(balanced_iemocap), file)


def load_transcription_embeddings_with_labels(transcriptions_path, sequence_len=30, embedding_size=400):
    transcriptions = json.load(open(transcriptions_path, "r"))
    labels = np.zeros(len(transcriptions))
    transcriptions_emb = np.zeros((len(transcriptions), sequence_len, embedding_size))
    for i, obj in enumerate(transcriptions):
        class_id = CLASS_TO_ID[obj["emotion"]]
        labels[i] = class_id
        preprocessed_transcription = Preprocessor.preprocess_one(obj["transcription"])
        transcriptions_emb[i] = Word2VecWrapper.get_sentence_embedding(preprocessed_transcription, sequence_len)
    return transcriptions_emb, labels


if __name__ == "__main__":
    mfcc_features, mfcc_labels = load_mfcc_dataset()
    train_features = mfcc_features[:VAL_SIZE]
    train_labels = mfcc_labels[:VAL_SIZE]
    val_features = mfcc_features[VAL_SIZE:]
    val_labels = mfcc_labels[VAL_SIZE:]
