import pickle
import json
import numpy as np

from word2vec_wrapper import Word2VecWrapper


IEMOCAP_PATH = "data/iemocap.pickle"
IEMOCAP_BALANCED_PATH = "data/iemocap_balanced.pickle"
TRANSCRIPTIONS_PATH = "data/iemocap_transcriptions.json"

USED_CLASSES = ["neu", "hap", "sad", "ang", "exc"]
CLASS_TO_ID = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}


def save_iemocap_transcriptions():
    """Load IEMOCAP and save transcriptions only in format: [{"emotion":E, "transcription": t},..] """
    iemocap = pickle.load(open(IEMOCAP_BALANCED_PATH, "rb"))
    transcriptions = [{"transcription": dic["transcription"], "emotion": dic["emotion"]} for dic in iemocap]
    with open(TRANSCRIPTIONS_PATH, "w") as file:
        json.dump(transcriptions, file)


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


def get_transcription_embeddings_and_labels(sequence_len=30, embedding_size=400):
    transcriptions = json.load(open(TRANSCRIPTIONS_PATH, "r"))
    labels = np.zeros((len(transcriptions), len(CLASS_TO_ID)))
    transcriptions_emb = np.zeros((len(transcriptions), sequence_len, embedding_size))
    num_exceeding = 0
    for i, obj in enumerate(transcriptions):
        class_id = CLASS_TO_ID[obj["emotion"]]
        labels[i][class_id] = 1
        if len(obj["transcription"].split(" ")) > sequence_len:
            num_exceeding += 1
        transcriptions_emb[i] = Word2VecWrapper.get_sentence_embedding(obj["transcription"], sequence_len)
    print(num_exceeding)
    print(labels.shape, transcriptions_emb.shape)


if __name__ == "__main__":
    get_transcription_embeddings_and_labels()