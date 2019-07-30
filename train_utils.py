import torch
import numpy as np

from torch.nn import functional as F

from confusion_matrix import ConfusionMatrix, get_error_ids


NUM_CLASSES = 4


def train(model, iterator, optimizer, criterion, reg_ratio):
    model.train()

    epoch_loss = 0
    conf_mat = ConfusionMatrix(np.zeros((NUM_CLASSES, NUM_CLASSES)))

    for batch, labels in iterator():
        optimizer.zero_grad()

        predictions = model(batch).squeeze(1)

        loss = criterion(predictions, labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + reg_ratio*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = epoch_loss / len(iterator)

    return average_loss, conf_mat


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    conf_mat = ConfusionMatrix(np.zeros((NUM_CLASSES, NUM_CLASSES)))

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch)

            loss = criterion(predictions.float(), labels)
            epoch_loss += loss.item()
            conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = epoch_loss / len(iterator)

    return average_loss, conf_mat


def train_ensemble(model, acoustic_iterator, linguistic_iterator, optimizer, criterion, reg_ratio):
    model.train()

    epoch_loss = 0
    conf_mat = ConfusionMatrix(np.zeros((NUM_CLASSES, NUM_CLASSES)))

    assert len(acoustic_iterator) == len(linguistic_iterator)

    for acoustic_tuple, linguistic_tuple in zip(acoustic_iterator(), linguistic_iterator()):
        acoustic_batch = acoustic_tuple[0]
        labels = acoustic_tuple[1]
        linguistic_batch = linguistic_tuple[0]
        optimizer.zero_grad()

        predictions = model(acoustic_batch, linguistic_batch).squeeze(1)

        loss = criterion(predictions, labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + reg_ratio*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = epoch_loss / len(acoustic_iterator)

    return average_loss, conf_mat


def eval_ensemble(ensemble_model, acoustic_model_iterator, linguistic_model_iterator, criterion):
    epoch_loss = 0
    conf_mat = ConfusionMatrix(np.zeros((NUM_CLASSES, NUM_CLASSES)))

    with torch.no_grad():
        for ((acoustic_batch, labels), (linguistic_batch, _)) in zip(acoustic_model_iterator(), linguistic_model_iterator()):
            predictions = ensemble_model(acoustic_batch, linguistic_batch)
            predictions = torch.Tensor(predictions)
            loss = criterion(predictions.float(), labels)
            epoch_loss += loss.item()
            conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = epoch_loss / len(acoustic_model_iterator)
    return average_loss, conf_mat