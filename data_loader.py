import pickle
from os.path import isfile, join

from word2vec_wrapper import Word2VecWrapper
from text_preprocessing import Preprocessor
from audio_preprocessing import *
from utils import timeit, log


IEMOCAP_PATH = "data/iemocap.pickle"
IEMOCAP_BALANCED_PATH = "data/iemocap_balanced.pickle"
IEMOCAP_BALANCED_ASR_PATH = "data/iemocap_balanced_asr.pickle"
IEMOCAP_FULL_PATH = "data/IEMOCAP_full_release"
LINGUISTIC_DATASET_PATH = "data/linguistic_features.npy"
LINGUISTIC_LABELS_PATH = "data/linguistic_labels.npy"
ACOUSTIC_FEATURES_PATH = "data/acoustic_features.npy"
ACOUSTIC_LABELS_PATH = "data/acoustic_labels.npy"
LINGUISTIC_DATASET_ASR_PATH = "data/linguistic_features_asr.npy"
LINGUISTIC_LABELS_ASR_PATH = "data/linguistic_labels_asr.npy"
SPECTROGRAMS_FEATURES_PATH = "data/spectrograms_features.npy"
SPECTROGRAMS_LABELS_PATH = "data/spectrograms_labels.npy"
MAPPING_ID_TO_SAMPLE_PATH =  "data/id_to_sample.json"
VAL_SIZE = 1531
USED_CLASSES = ["neu", "hap", "sad", "ang", "exc"]
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
ID_TO_FULL_CLASS = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Anger"}
LAST_SESSION_SAMPLE_ID = 4290


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


def split_dataset_session_wise(dataset_features, dataset_labels, split_ratio=0.1):
    """Splittng dataset into train/val sets by taking every nth sample to val set"""

    test_indexes = list(range(LAST_SESSION_SAMPLE_ID, dataset_features.shape[0]))

    skip_ratio = int(1/split_ratio)
    all_indexes = list(range(dataset_features.shape[0]))
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
    assert test_features.shape[0] + val_features.shape[0] + train_features.shape[0] == dataset_features.shape[0]
    return test_features, test_labels, val_features, val_labels, train_features, train_labels


def load_or_create_dataset(create_func, features_path, labels_path, **kwargs):
    """Extracting & Saving dataset"""
    if not isfile(features_path) or not isfile(labels_path):
        print("Dataset not found. Creating dataset...")
        create_func(**kwargs)
        print("Dataset created. Loading dataset...")

    """Loading dataset"""
    dataset = np.load(features_path)
    labels = np.load(labels_path)
    print("Dataset loaded.")

    assert dataset.shape[0] == labels.shape[0]

    return split_dataset_session_wise(dataset, labels)


@timeit
def create_spectrogram_dataset(**kwargs):
    with open(IEMOCAP_BALANCED_PATH, 'rb') as handle:
        iemocap = pickle.load(handle)

    labels = []
    spectrograms = []

    for i, sample in enumerate(iemocap):
        labels.append(CLASS_TO_ID[sample['emotion']])
        session_id = sample['id'].split('Ses0')[1][0]
        sample_dir = "_".join(sample['id'].split("_")[:-1])
        sample_name = "{}.wav".format(sample['id'])
        abs_path = join(IEMOCAP_FULL_PATH, "Session{}".format(session_id), "sentences/wav/", sample_dir, sample_name)
        spectrogram = generate_spectrogram(abs_path, kwargs.get("view", False))
        spectrograms.append(spectrogram)
        if (i % 100 == 0):
            print(i)

    np.save(SPECTROGRAMS_LABELS_PATH, np.array(labels))
    np.save(SPECTROGRAMS_FEATURES_PATH, np.array(spectrograms))


@timeit
def create_acoustic_dataset(**kwargs):
    framerate = kwargs.get("framerate", 16000)

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
def create_linguistic_dataset(**kwargs):
    asr = kwargs.get("asr", False)
    sequence_len = kwargs.get("sequence_len", 30)
    embedding_size = kwargs.get("embedding_size", 400)
    iemocap = pickle.load(open(IEMOCAP_BALANCED_ASR_PATH, "rb"))

    labels = np.zeros(len(iemocap))
    transcriptions_emb = np.zeros((len(iemocap), sequence_len, embedding_size))

    for i, obj in enumerate(iemocap):
        class_id = CLASS_TO_ID[obj["emotion"]]
        labels[i] = class_id
        transcription = obj["asr_transcription"] if asr else obj["transcription"]
        preprocessed_transcription = Preprocessor.preprocess_one(transcription)
        transcriptions_emb[i] = Word2VecWrapper.get_sentence_embedding(preprocessed_transcription, sequence_len)

    dataset_path = LINGUISTIC_DATASET_ASR_PATH if asr else LINGUISTIC_DATASET_PATH
    labels_path = LINGUISTIC_LABELS_ASR_PATH if asr else LINGUISTIC_LABELS_PATH

    np.save(dataset_path, transcriptions_emb)
    np.save(labels_path, labels)


@timeit
def load_spectrogram_dataset():
    return load_or_create_dataset(create_spectrogram_dataset, SPECTROGRAMS_FEATURES_PATH, SPECTROGRAMS_LABELS_PATH)


@timeit
def load_acoustic_features_dataset():
    return load_or_create_dataset(create_acoustic_dataset, ACOUSTIC_FEATURES_PATH, ACOUSTIC_LABELS_PATH)


@timeit
def load_linguistic_dataset(asr=False):
    dataset_path = LINGUISTIC_DATASET_ASR_PATH if asr else LINGUISTIC_DATASET_PATH
    labels_path = LINGUISTIC_LABELS_ASR_PATH if asr else LINGUISTIC_LABELS_PATH
    return load_or_create_dataset(create_linguistic_dataset, dataset_path, labels_path, asr=asr)


@timeit
def generate_transcriptions():
    from deepspeech_generator import speech_to_text
    print("Loading iemocap...")
    iemocap = pickle.load(open(IEMOCAP_BALANCED_PATH, "rb"))
    print("Done. Generating transcriptions...")
    for i, sample in enumerate(iemocap):
        session_id = sample['id'].split('Ses0')[1][0]
        sample_dir = "_".join(sample['id'].split("_")[:-1])
        sample_name = "{}.wav".format(sample['id'])
        abs_path = join(IEMOCAP_FULL_PATH, "Session{}".format(session_id), "sentences/wav/", sample_dir, sample_name)
        transcription = speech_to_text("models/output_graph.pbmm", "models/alphabet.txt", "models/lm.binary", "models/trie", abs_path)
        iemocap[i]["asr_transcription"] = transcription
        if not i % 10 and i != 0:
            log("{}/{}".format(i, len(iemocap)), True)
    pickle.dump(iemocap, open(IEMOCAP_BALANCED_ASR_PATH, "wb"))
