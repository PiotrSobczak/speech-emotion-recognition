import torch
import numpy as np

from metrics import accuracy, confusion_matrix


NUM_CLASSES = 4


def train(model, iterator, optimizer, criterion, reg_ratio):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch, labels in iterator():
        optimizer.zero_grad()

        predictions = model(batch).squeeze(1)

        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + reg_ratio*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch).squeeze(1)

            loss = criterion(predictions.float(), labels)

            acc = accuracy(predictions, labels)
            conf_mat += confusion_matrix(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator), conf_mat