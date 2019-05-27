import torch
import numpy as np


class BatchIterator:
    def __init__(self, dataset, labels, batch_size=50, embedding_size=400):
        self._dataset = dataset
        self._size = len(dataset)
        self._labels = labels
        self._batch_size = batch_size
        self._embedding_size = embedding_size

    def __len__(self):
        return int(self._size/self._batch_size)

    def __call__(self):
        num_batches = int(self._size/self._batch_size)
        for i in range(num_batches):
            input_batch = self._dataset[i*self._batch_size:(i+1)*self._batch_size]
            labels_batch = self._labels[i*self._batch_size:(i+1)*self._batch_size]
            yield input_batch, torch.LongTensor(labels_batch)

    def shuffle(self):
        new_order = np.random.permutation(self._size)
        self._dataset = self._dataset[new_order]
        self._labels = self._labels[new_order]
        
