import torch
import numpy as np

from ser.config import NUM_CLASSES


class ConfusionMatrix:
    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    def __add__(self, other):
        confusion_matrix_sum = self.confusion_matrix + other.confusion_matrix
        return ConfusionMatrix(confusion_matrix_sum)

    def __str__(self):
        return self.confusion_matrix.__str__()

    @staticmethod
    def from_predictions(predictions, labels):
        """ Returns confusion matrix """
        assert predictions.shape[0] == labels.shape[0], "1 dim of predictions and labels must be equal"
        rounded_preds = torch.argmax(predictions, 1)
        conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for i in range(rounded_preds.shape[0]):
            predicted_class = rounded_preds[i]
            correct_class = labels[i]
            conf_mat[correct_class][predicted_class] += 1
        return ConfusionMatrix(conf_mat)

    @property
    def size(self):
        return self.confusion_matrix.shape[0]

    @property
    def accuracy(self):
        """ Returns accuracy"""
        return sum([self.confusion_matrix[i, i] for i in range(self.size)]) / self.confusion_matrix.sum()

    @property
    def unweighted_accuracy(self):
        """ Returns unweighted accuracy, also called unweighted recall"""
        uacc_per_class = [self.confusion_matrix[i, i]/self.confusion_matrix[i].sum() for i in range(self.size)]
        return sum(uacc_per_class)/len(uacc_per_class)

    @property
    def average_precision(self):
        """ Returns unweighted accuracy, also called unweighted recall"""
        prec_per_class = [self.confusion_matrix[i, i] / self.confusion_matrix[:, i].sum() for i in range(self.size)]
        return sum(prec_per_class)/len(prec_per_class)
