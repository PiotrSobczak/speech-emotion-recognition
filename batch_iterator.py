from word2vec_wrapper import Word2VecWrapper
import torch
import numpy as np


class BatchIterator:
    def __init__(self, raw_batch_list, sequence_len=30, batch_size=64, embedding_size=400):
        self._raw_batch_list = raw_batch_list
        self._size = len(raw_batch_list)
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._sequence_len = sequence_len

    def __len__(self):
        return self._size

    def __call__(self):
        for raw_batch in self._raw_batch_list[1:]:
            yield self._create_batch(raw_batch)

    def _create_batch(self, raw_batch):
        batch = np.zeros((self._batch_size, self._sequence_len, self._embedding_size))
        for sentence_id, sentence in enumerate(raw_batch["inputs"]):
            batch[sentence_id] = Word2VecWrapper.get_sentence_embedding(sentence, self._sequence_len)

        return batch.swapaxes(0, 1), torch.tensor(np.array(raw_batch["labels"])).float()