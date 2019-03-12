import pickle
import numpy as np


class Word2VecWrapper:
    is_init=False
    word_to_index = None
    embedding_array = None
    EMBEDDING_SIZE = 400

    @classmethod
    def init(cls):
        if not cls.is_init:
            cls.embedding_array = np.load("data/embeddings_array.numpy")
            cls.word_to_index = pickle.load(open("data/word_to_index.pickle", "rb"))
            cls.is_init = True
            print("Initialized Word2VecWrapper")
        else:
            print("Word2VecWrapper already initialized")

    @classmethod
    def vocab_contains(cls, word):
        if not cls.is_init:
            cls.init()
        return word in cls.word_to_index

    @classmethod
    def get_embedding(cls, word):
        if cls.is_init:
            if word in cls.word_to_index:
                return cls.embedding_array[cls.word_to_index[word]]
            else:
                return np.zeros((1, cls.EMBEDDING_SIZE))
        else:
            cls.init()
            return cls.get_embedding(word)

    @classmethod
    def get_sentence_embedding(cls, sentence, sequence_len):
        words = sentence.split(" ")
        sentence_embedded = np.zeros((sequence_len, cls.EMBEDDING_SIZE))
        for word_id in range(sequence_len):
            if word_id < len(words):
                sentence_embedded[word_id] = cls.get_embedding(words[word_id])
        return sentence_embedded