import torch
import numpy as np
from config import NUM_CLASSES
from confusion_matrix import ConfusionMatrix
from models import WeightedAverageEnsemble


def run_epoch_train(model, iterator, optimizer, criterion, reg_ratio):
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


def run_epoch_eval(model, iterator, criterion):
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


def search_for_optimal_alpha(acoustic_model, linguistic_model, val_iter_ensemble):
    print("Searching for the optimal alpha...")
    alphas = {}
    for alpha in np.linspace(0.01, 0.99, 49):
        weightedAverageEnsemble = WeightedAverageEnsemble(acoustic_model, linguistic_model, alpha)
        _, val_cm = run_epoch_eval(weightedAverageEnsemble, val_iter_ensemble, torch.nn.CrossEntropyLoss())
        alphas[alpha] = val_cm.accuracy
    max_val = max(alphas.values())
    max_val_id = list(alphas.values()).index(max_val)
    max_val_alpha = list(alphas.keys())[max_val_id]
    assert alphas[max_val_alpha] == max_val
    print("Found optimal alpha={}".format(max_val_alpha))
    return max_val_alpha