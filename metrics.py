import torch
import numpy as np
import json

from data_loader import ID_TO_CLASS, MAPPING_ID_TO_SAMPLE_PATH

NUM_CLASSES = 4


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


def get_error_ids(preds, y, offset=0):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds, 1)
    error_ids = {}
    mapping_json = json.load(open(MAPPING_ID_TO_SAMPLE_PATH, "r"))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        if predicted_class != correct_class:
            error_ids[i + offset] = {"output": ID_TO_CLASS[int(predicted_class)], "label": ID_TO_CLASS[int(correct_class)]}
            error_ids[i + offset].update(mapping_json[str(i + offset)])
    return error_ids