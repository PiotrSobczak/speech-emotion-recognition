import torch
import numpy as np

from torch.nn import functional as F

from metrics import confusion_matrix, get_error_ids


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
    uacc_per_class = [conf_mat[i, i] / conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    unweighted_acc = sum(uacc_per_class) / len(uacc_per_class)

    return epoch_loss / len(iterator), acc, unweighted_acc, conf_mat


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch)

            loss = criterion(predictions.float(), labels)
            epoch_loss += loss.item()
            conf_mat += confusion_matrix(predictions, labels)

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    uacc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    unweighted_acc = sum(uacc_per_class)/len(uacc_per_class)

    prec_per_class = [conf_mat[i, i] / conf_mat[:, i].sum() for i in range(conf_mat.shape[0])]
    average_precision = sum(prec_per_class)/len(prec_per_class)

    return epoch_loss / len(iterator), acc, unweighted_acc, average_precision, conf_mat


def eval_decision_ensemble(acoustic_model, linguistic_model, acoustic_model_iterator, linguistic_model_iterator, criterion, ensemble_type, alpha=0.5):
    acoustic_model.eval()
    linguistic_model.eval()

    epoch_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        for ((acoustic_batch, labels), (linguistic_batch, _)) in zip(acoustic_model_iterator(), linguistic_model_iterator()):
            predictions_acoustic = F.log_softmax(acoustic_model(acoustic_batch).squeeze(1), dim=1)
            predictions_linguistic = F.log_softmax(linguistic_model(linguistic_batch).squeeze(1), dim=1)

            if ensemble_type == "average":
                predictions = (predictions_acoustic + predictions_linguistic) / 2
            elif ensemble_type == "w_avg":
                predictions = predictions_acoustic * alpha + predictions_linguistic * (1-alpha)
            elif ensemble_type == "confidence":
                predictions = np.zeros(predictions_acoustic.shape)
                for i in range(predictions_acoustic.shape[0]):
                    predictions[i] = predictions_acoustic[i] if predictions_acoustic[i].max() > predictions_linguistic[i].max() else predictions_linguistic[i]
            predictions = torch.Tensor(predictions)
            loss = criterion(predictions.float(), labels)
            epoch_loss += loss.item()
            conf_mat += confusion_matrix(predictions, labels)

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    uacc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    unweighted_acc = sum(uacc_per_class)/len(uacc_per_class)

    prec_per_class = [conf_mat[i, i] / conf_mat[:, i].sum() for i in range(conf_mat.shape[0])]
    average_precision = sum(prec_per_class)/len(prec_per_class)

    return epoch_loss / len(acoustic_model_iterator), acc, unweighted_acc, average_precision, conf_mat


def train_ensemble(model, acoustic_iterator, linguistic_iterator, optimizer, criterion, reg_ratio):
    model.train()

    epoch_loss = 0
    conf_mat = np.zeros((4, 4))

    assert len(acoustic_iterator) == len(linguistic_iterator)

    for acoustic_tuple, linguistic_tuple in zip(acoustic_iterator(), linguistic_iterator()):
        acoustic_batch = acoustic_tuple[0]
        acoustic_labels = acoustic_tuple[1]
        linguistic_batch = linguistic_tuple[0]
        linguistic_labels = linguistic_tuple[1]
        optimizer.zero_grad()

        predictions = model(acoustic_batch, linguistic_batch).squeeze(1)

        loss = criterion(predictions, acoustic_labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + reg_ratio*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        conf_mat += confusion_matrix(predictions, acoustic_labels)

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])]) / conf_mat.sum()
    uacc_per_class = [conf_mat[i, i] / conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    unweighted_acc = sum(uacc_per_class) / len(uacc_per_class)

    return epoch_loss / len(acoustic_iterator), acc, unweighted_acc, conf_mat


def eval_feature_ensemble(model, acoustic_iterator, linguistic_iterator, criterion, global_offset=0):
    model.eval()

    epoch_loss = 0
    conf_mat = np.zeros((4, 4))

    assert len(acoustic_iterator) == len(linguistic_iterator)

    with torch.no_grad():
        error_ids={}
        for i, (acoustic_tuple, linguistic_tuple) in enumerate(zip(acoustic_iterator(), linguistic_iterator())):
            acoustic_batch = acoustic_tuple[0]
            acoustic_labels = acoustic_tuple[1]
            linguistic_batch = linguistic_tuple[0]
            linguistic_labels = linguistic_tuple[1]
            predictions = model(acoustic_batch, linguistic_batch).squeeze(1)

            loss = criterion(predictions.float(), acoustic_labels)
            epoch_loss += loss.item()
            conf_mat += confusion_matrix(predictions, acoustic_labels)
            error_ids.update(get_error_ids(predictions, acoustic_labels, global_offset+i*acoustic_iterator._batch_size))

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    uacc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    unweighted_acc = sum(uacc_per_class)/len(uacc_per_class)

    prec_per_class = [conf_mat[i, i] / conf_mat[:, i].sum() for i in range(conf_mat.shape[0])]
    average_precision = sum(prec_per_class)/len(prec_per_class)

    return epoch_loss / len(acoustic_iterator), acc, unweighted_acc, average_precision, conf_mat, error_ids