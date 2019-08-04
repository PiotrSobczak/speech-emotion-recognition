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

    def shuffle(self, order=None):
        if order is None:
            order = np.random.permutation(self._size)
        self._dataset = self._dataset[order]
        self._labels = self._labels[order]


class EnsembleBatchIterator:
    def __init__(self, acoustic_iterator, linguistic_iterator):
        self._linguistic_iterator = linguistic_iterator
        self._acoustic_iterator = acoustic_iterator
        assert self._linguistic_iterator._size == self._acoustic_iterator._size, "Inconsistent iterator sizes"
        assert self._linguistic_iterator.__len__() == self._acoustic_iterator.__len__(), "Inconsistent iterator len"
        self._size = self._linguistic_iterator._size
        self._batch_size = 100

    def __call__(self):
        for acoustic_tuple, linguistic_tuple in zip(self._acoustic_iterator(), self._linguistic_iterator()):
            acoustic_batch, acoustic_labels = acoustic_tuple
            linguistic_batch, linguistic_labels = linguistic_tuple
            yield (acoustic_batch, linguistic_batch), acoustic_labels

    def __len__(self):
        return int(self._size/self._batch_size)

    def shuffle(self):
        order = np.random.permutation(self._linguistic_iterator._size)
        self._linguistic_iterator.shuffle(order)
        self._acoustic_iterator.shuffle(order)