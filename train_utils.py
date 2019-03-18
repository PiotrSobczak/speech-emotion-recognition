import torch
import numpy as np

from metrics import confusion_matrix


NUM_CLASSES = 4


def train(model, iterator, optimizer, criterion, reg_ratio):
    model.train()

    epoch_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

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

        conf_mat += confusion_matrix(predictions, labels)

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])]) / conf_mat.sum()
    acc_per_class = [conf_mat[i, i] / conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    weighted_acc = sum(acc_per_class) / len(acc_per_class)

    return epoch_loss / len(iterator), acc, weighted_acc, conf_mat


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch).squeeze(1)

            loss = criterion(predictions.float(), labels)
            epoch_loss += loss.item()
            conf_mat += confusion_matrix(predictions, labels)

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    acc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    weighted_acc = sum(acc_per_class)/len(acc_per_class)

    return epoch_loss / len(iterator), acc, weighted_acc, conf_mat