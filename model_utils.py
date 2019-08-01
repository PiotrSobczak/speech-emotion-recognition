import torch
import numpy as np
from config import NUM_CLASSES
from confusion_matrix import ConfusionMatrix
from models import WeightedAverageEnsemble


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
    epoch_losses = []
    conf_mat = ConfusionMatrix(np.zeros((NUM_CLASSES, NUM_CLASSES)))

    with torch.no_grad():
        for ((acoustic_batch, labels), (linguistic_batch, _)) in zip(acoustic_model_iterator(), linguistic_model_iterator()):
            predictions = ensemble_model(acoustic_batch, linguistic_batch)
            predictions = torch.Tensor(predictions)
            loss = criterion(predictions.float(), labels)
            epoch_losses.append(loss.item())
            conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

    average_loss = sum(epoch_losses) / len(acoustic_model_iterator)
    return average_loss, conf_mat


def search_for_optimal_alpha(acoustic_model, linguistic_model, val_iter_acoustic, val_iter_linguistic):
    print("Searching for the optimal alpha...")
    alphas = {}
    for alpha in np.linspace(0.01, 0.99, 49):
        weightedAverageEnsemble = WeightedAverageEnsemble(acoustic_model, linguistic_model, alpha)
        _, val_cm = eval_ensemble(
            weightedAverageEnsemble, val_iter_acoustic, val_iter_linguistic, torch.nn.CrossEntropyLoss()
        )
        alphas[alpha] = val_cm.accuracy
    max_val = max(alphas.values())
    max_val_id = list(alphas.values()).index(max_val)
    max_val_alpha = list(alphas.keys())[max_val_id]
    assert alphas[max_val_alpha] == max_val
    print("Found optimal alpha={}".format(max_val_alpha))
    return max_val_alpha