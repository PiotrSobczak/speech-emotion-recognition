import torch
import numpy as np

NUM_CLASSES = 4


def accuracy(preds, y):
    """ Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8 """
    rounded_preds = torch.argmax(preds,1)
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc


def confusion_matrix(preds, y):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds,1)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        conf_mat[correct_class][predicted_class] += 1
    return conf_mat