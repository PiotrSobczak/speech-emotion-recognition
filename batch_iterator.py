import torch
import numpy as np


class BatchIterator:
    def __init__(self, transcriptions, labels, batch_size=50, embedding_size=400):
        self._transcriptions = transcriptions
        self._size = len(transcriptions)
        self._labels = labels
        self._batch_size = batch_size
        self._embedding_size = embedding_size

    def __len__(self):
        return int(self._size/self._batch_size)

    def __call__(self):
        num_batches = int(self._size/self._batch_size)
        for i in range(num_batches):
            transcriptions_batch = self._transcriptions[i*self._batch_size:(i+1)*self._batch_size]
            labels_batch = self._labels[i*self._batch_size:(i+1)*self._batch_size]
            yield transcriptions_batch.swapaxes(0, 1), torch.cuda.LongTensor(labels_batch)

    def shuffle(self):
        new_order = np.random.permutation(self._size)
        self._transcriptions = self._transcriptions[new_order]
        self._labels = self._labels[new_order]
        
