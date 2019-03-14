import torch


class BatchIterator:
    def __init__(self, transcriptions, labels, sequence_len=30, batch_size=50, embedding_size=400):
        self._transcriptions = transcriptions
        self._size = len(transcriptions)
        self._labels = labels
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._sequence_len = sequence_len

    def __len__(self):
        return int(self._size/self._batch_size)

    def __call__(self):
        num_batches = int(self._size/self._batch_size)
        for i in range(num_batches):
            transcriptions_batch = self._transcriptions[i*self._batch_size:(i+1)*self._batch_size]
            labels_batch = self._labels[i*self._batch_size:(i+1)*self._batch_size]
            yield transcriptions_batch.swapaxes(0, 1), torch.tensor(labels_batch).long()