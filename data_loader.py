import pickle
import numpy as np
from os.path import isfile
from python_speech_features import mfcc

from word2vec_wrapper import Word2VecWrapper
from preprocessing import Preprocessor
from utils import timeit
from iemocap_utils.features import stFeatureExtraction

IEMOCAP_PATH = "data/iemocap.pickle"
IEMOCAP_BALANCED_PATH = "data/iemocap_balanced.pickle"

LINGUISTIC_DATASET_PATH = "data/linguistic_features.npy"
LINGUISTIC_LABELS_PATH = "data/linguistic_labels.npy"
ACOUSTIC_FEATURES_PATH = "data/acoustic_features.npy"
ACOUSTIC_LABELS_PATH = "data/acoustic_labels.npy"

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
def create_acoustic_dataset_mfcc(num_mfcc_features=26, window_len=0.025, winstep=0.01, sample_rate=16000, seq_len=800):
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


def split_dataset_skip(dataset_features, dataset_labels, split_ratio=0.2):
    """Splittng dataset into train/val sets by taking every nth sample to val set"""
    skip_ratio = int(1/split_ratio)
    all_indexes = list(range(dataset_features.shape[0]))
    test_indexes = list(range(0, dataset_features.shape[0], skip_ratio))
    train_indexes = list(set(all_indexes) - set(test_indexes))
    val_indexes = train_indexes[::skip_ratio]
    train_indexes = list(set(train_indexes) - set(val_indexes))

    test_features = dataset_features[test_indexes]
    test_labels = dataset_labels[test_indexes]
    val_features = dataset_features[val_indexes]
    val_labels = dataset_labels[val_indexes]
    train_features = dataset_features[train_indexes]
    train_labels = dataset_labels[train_indexes]

    assert test_features.shape[0] == test_labels.shape[0]
    assert val_features.shape[0] == val_labels.shape[0]
    assert train_features.shape[0] == train_labels.shape[0]

    return test_features, test_labels, val_features, val_labels, train_features, train_labels


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
def create_acoustic_dataset(framerate=16000):
    def calculate_acoustic_features(frames, freq, options):
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

    with open(IEMOCAP_BALANCED_PATH, 'rb') as handle:
        iemocap = pickle.load(handle)

    acoustic_labels = []
    acoustic_features = []

    for i, ses_mod in enumerate(iemocap):
        acoustic_labels.append(CLASS_TO_ID[ses_mod['emotion']])
        x_head = ses_mod['signal']
        st_features = calculate_acoustic_features(x_head, framerate, None)
        st_features, _ = pad_sequence_into_array(st_features, maxlen=200)
        acoustic_features.append(st_features.T)
        if (i % 100 == 0):
            print(i)

    np.save(ACOUSTIC_LABELS_PATH, np.array(acoustic_labels))
    np.save(ACOUSTIC_FEATURES_PATH, np.array(acoustic_features))

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

    return split_dataset_skip(linguistic_dataset, linguistic_labels)


def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):

    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask


if __name__ == "__main__":
    val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
