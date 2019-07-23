import torch
import numpy as np
from math import ceil


class BatchIterator:
    DEFAULT_EVALUATE_BATCH_SIZE = 100

    def __init__(self, dataset, labels, batch_size=DEFAULT_EVALUATE_BATCH_SIZE):
        self._dataset = dataset
        self._size = len(dataset)
        self._labels = labels
        self._batch_size = batch_size
        self._label_tensor_type = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self._input_tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def __len__(self):
        return int(self._size/self._batch_size)

    def __call__(self):
        num_batches = ceil(self._size/self._batch_size)
        for i in range(num_batches):
            input_batch = self._dataset[i*self._batch_size:(i+1)*self._batch_size]
            labels_batch = self._labels[i*self._batch_size:(i+1)*self._batch_size]
            yield self._input_tensor_type(input_batch), self._label_tensor_type(labels_batch)

    def shuffle(self):
        new_order = np.random.permutation(self._size)
        self._dataset = self._dataset[new_order]
        self._labels = self._labels[new_order]

