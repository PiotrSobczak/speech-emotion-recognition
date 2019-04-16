import pickle
import numpy as np
from os.path import isfile
from python_speech_features import mfcc

from word2vec_wrapper import Word2VecWrapper
from preprocessing import Preprocessor
from utils import timeit

IEMOCAP_PATH = "data/iemocap.pickle"
IEMOCAP_BALANCED_PATH = "data/iemocap_balanced.pickle"

LINGUISTIC_DATASET_PATH = "data/linguistic_features.npy"
LINGUISTIC_LABELS_PATH = "data/linguistic_labels.npy"
ACOUSTIC_FEATURES_PATH = "data/mfcc_features.npy"
ACOUSTIC_LABELS_PATH = "data/mfcc_labels.npy"

VAL_SIZE = 1531
USED_CLASSES = ["neu", "hap", "sad", "ang", "exc"]
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}


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


@timeit
def create_acoustic_dataset(num_mfcc_features=26, window_len=0.025, winstep=0.01, sample_rate=16000, seq_len=800):
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
        mfcc_features = mfcc(obj["signal"], sample_rate, window_len, winstep, num_mfcc_features)
        mfcc_features = mfcc_features[:seq_len]
        mfcc_features_dataset[i, :mfcc_features.shape[0], :mfcc_features.shape[1]] = mfcc_features

    np.save(ACOUSTIC_FEATURES_PATH, mfcc_features_dataset)
    np.save(ACOUSTIC_LABELS_PATH, labels)

@timeit
def normalize_dataset(mfcc_features):
    averages = np.zeros((mfcc_features.shape[0], mfcc_features.shape[1], mfcc_features.shape[2]))
    ranges = np.zeros((mfcc_features.shape[0], mfcc_features.shape[1], mfcc_features.shape[2]))

    for i in range(mfcc_features.shape[0]):
        averages[i, :, :] = (mfcc_features[i].max() + mfcc_features[i].min()) / 2
        ranges[i, :, :] = (mfcc_features[i].max() - mfcc_features[i].min()) / 2

    """NOTE, skipping subtracking average in order to leave zeros. Try normalization before zero-padding."""
    return mfcc_features / ranges


def split_dataset_skip(dataset_features, dataset_labels):
    """Splittng dataset into train/val sets by taking every nth sample to val set"""
    skip_ratio = int(dataset_features.shape[0]/VAL_SIZE) + 1
    all_indexes = list(range(dataset_features.shape[0]))
    val_indexes = list(range(0, dataset_features.shape[0], skip_ratio))
    train_indexes = list(set(all_indexes) - set(val_indexes))

    val_features = dataset_features[val_indexes]
    val_labels = dataset_labels[val_indexes]
    train_features = dataset_features[train_indexes]
    train_labels = dataset_labels[train_indexes]

    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]

    return val_features, val_labels, train_features, train_labels


def split_dataset_head(dataset_features, dataset_labels):
    """Splittng dataset into train/val sets by taking n first samples to val set"""
    val_features = dataset_features[:VAL_SIZE]
    val_labels = dataset_labels[:VAL_SIZE]
    train_features = dataset_features[VAL_SIZE:]
    train_labels = dataset_labels[VAL_SIZE:]

    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]

    return val_features, val_labels, train_features, train_labels

@timeit
def load_acoustic_dataset():
    """Extracting & Saving dataset"""
    if not isfile(ACOUSTIC_FEATURES_PATH) or not isfile(ACOUSTIC_LABELS_PATH):
        print("Acoustic dataset not found. Creating dataset...")
        create_acoustic_dataset()
        print("Acoustic dataset created. Loading dataset...")

    """Loading acoustic dataset"""
    mfcc_features = np.load(ACOUSTIC_FEATURES_PATH)
    mfcc_labels = np.load(ACOUSTIC_LABELS_PATH)
    print("Acoustic dataset loaded.")

    assert mfcc_features.shape[0] == mfcc_labels.shape[0]

    return split_dataset_skip(mfcc_features, mfcc_labels)


@timeit
def create_linguistic_dataset(sequence_len=30, embedding_size=400):
    iemocap = pickle.load(open(IEMOCAP_BALANCED_PATH, "rb"))
    transcriptions = [{"transcription": dic["transcription"], "emotion": dic["emotion"]} for dic in iemocap]
    labels = np.zeros(len(transcriptions))
    transcriptions_emb = np.zeros((len(transcriptions), sequence_len, embedding_size))
    for i, obj in enumerate(transcriptions):
        class_id = CLASS_TO_ID[obj["emotion"]]
        labels[i] = class_id
        preprocessed_transcription = Preprocessor.preprocess_one(obj["transcription"])
        transcriptions_emb[i] = Word2VecWrapper.get_sentence_embedding(preprocessed_transcription, sequence_len)

    np.save(LINGUISTIC_DATASET_PATH, transcriptions_emb)
    np.save(LINGUISTIC_LABELS_PATH, labels)


@timeit
def load_linguistic_dataset():
    """Extracting & Saving dataset"""
    if not isfile(LINGUISTIC_DATASET_PATH) or not isfile(LINGUISTIC_LABELS_PATH):
        print("Linguistic dataset not found. Creating dataset...")
        create_linguistic_dataset()
        print("Linguistic dataset created. Loading dataset...")

    """Loading linguistic dataset"""
    linguistic_dataset = np.load(LINGUISTIC_DATASET_PATH)
    linguistic_labels = np.load(LINGUISTIC_LABELS_PATH)
    print("Linguistic dataset loaded.")

    assert linguistic_dataset.shape[0] == linguistic_labels.shape[0]

    return split_dataset_head(linguistic_dataset, linguistic_labels)


if __name__ == "__main__":
    val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
