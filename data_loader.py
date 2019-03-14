import pickle
import json
import numpy as np

from word2vec_wrapper import Word2VecWrapper


IEMOCAP_PATH = "data/iemocap.pickle"
IEMOCAP_BALANCED_PATH = "data/iemocap_balanced.pickle"
TRANSCRIPTIONS_PATH = "data/iemocap_transcriptions.json"
TRANSCRIPTIONS_VAL_PATH = "data/iemocap_transcriptions_val.json"
TRANSCRIPTIONS_TRAIN_PATH = "data/iemocap_transcriptions_train.json"

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


def get_transcription_embeddings_and_labels(transcriptions_path, sequence_len=30, embedding_size=400):
    transcriptions = json.load(open(transcriptions_path, "r"))
    # labels = np.zeros((len(transcriptions), len(CLASS_TO_ID)))
    labels = np.zeros(len(transcriptions))
    transcriptions_emb = np.zeros((len(transcriptions), sequence_len, embedding_size))
    for i, obj in enumerate(transcriptions):
        class_id = CLASS_TO_ID[obj["emotion"]]
        # labels[i][class_id] = 1
        labels[i] = class_id
        transcriptions_emb[i] = Word2VecWrapper.get_sentence_embedding(obj["transcription"], sequence_len)
    return transcriptions_emb, labels


if __name__ == "__main__":
    transcriptions_emb_val, labels_val = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_VAL_PATH)
    print("Loaded val data: {}, {}".format(labels_val.shape, transcriptions_emb_val.shape))
    transcriptions_emb_train, labels_train = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_TRAIN_PATH)
    print("Loaded train data: {}, {}".format(labels_train.shape, transcriptions_emb_train.shape))