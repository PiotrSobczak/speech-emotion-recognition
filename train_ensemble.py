import torch
import os
from time import gmtime, strftime
from metrics import confusion_matrix
import json
import argparse
import numpy as np

from models import AttentionModel, CNN, EnsembleModel
from batch_iterator import BatchIterator
from data_loader import load_linguistic_dataset, load_spectrogram_dataset
from utils import log, log_major, log_success
from config import LinguisticConfig, AcousticSpectrogramConfig as AcousticConfig

MODEL_PATH = "saved_models"


def train(model, acoustic_iterator, linguistic_iterator, optimizer, criterion, reg_ratio):
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
    acc_per_class = [conf_mat[i, i] / conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    weighted_acc = sum(acc_per_class) / len(acc_per_class)

    return epoch_loss / len(acoustic_iterator), acc, weighted_acc, conf_mat


def evaluate(model, acoustic_iterator, linguistic_iterator, criterion):
    model.eval()

    epoch_loss = 0
    conf_mat = np.zeros((4, 4))

    assert len(acoustic_iterator) == len(linguistic_iterator)

    with torch.no_grad():
        for acoustic_tuple, linguistic_tuple in zip(acoustic_iterator(), linguistic_iterator()):
            acoustic_batch = acoustic_tuple[0]
            acoustic_labels = acoustic_tuple[1]
            linguistic_batch = linguistic_tuple[0]
            linguistic_labels = linguistic_tuple[1]
            predictions = model(acoustic_batch, linguistic_batch).squeeze(1)

            loss = criterion(predictions.float(), acoustic_labels)
            epoch_loss += loss.item()
            conf_mat += confusion_matrix(predictions, acoustic_labels)

    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    acc_per_class = [conf_mat[i, i]/conf_mat[i].sum() for i in range(conf_mat.shape[0])]
    weighted_acc = sum(acc_per_class)/len(acc_per_class)

    return epoch_loss / len(acoustic_iterator), acc, weighted_acc, conf_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--linguistic_model", type=str, required=True)
    parser.add_argument("-a", "--acoustic_model", type=str, required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.acoustic_model), "acoustic_model weights file does not exist"
    assert os.path.isfile(args.acoustic_model.replace(".torch", ".json")), "acoustic_model config file does not exist"
    assert os.path.isfile(args.linguistic_model), "linguistic_model weights file does not exist"
    assert os.path.isfile(args.linguistic_model.replace(".torch", ".json")), "linguistic_model config file does not exist"

    test_features_acoustic, test_labels_acoustic, val_features_acoustic, val_labels_acoustic, train_features_acoustic, train_labels_acoustic = load_spectrogram_dataset()
    test_features_linguistic, test_labels_linguistic, val_features_linguistic, val_labels_linguistic, train_features_linguistic, train_labels_linguistic = load_linguistic_dataset()

    test_iterator_acoustic = BatchIterator(test_features_acoustic, test_labels_acoustic, 100)
    test_iterator_linguistic = BatchIterator(test_features_linguistic, test_labels_linguistic, 100)

    val_iterator_acoustic = BatchIterator(val_features_acoustic, val_labels_acoustic, 100)
    val_iterator_linguistic = BatchIterator(val_features_linguistic, val_labels_linguistic, 100)

    train_iterator_acoustic = BatchIterator(train_features_acoustic, train_labels_acoustic, 100)
    train_iterator_linguistic = BatchIterator(train_features_linguistic, train_labels_linguistic, 100)

    assert np.array_equal(test_labels_acoustic,
                          test_labels_linguistic), "Labels for acoustic and linguistic datasets are not the same!"

    """Choosing hardware"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        print("Using GPU. Setting default tensor type to torch.cuda.FloatTensor")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("Using CPU. Setting default tensor type to torch.FloatTensor")
        torch.set_default_tensor_type("torch.FloatTensor")

    """Converting model to specified hardware and format"""
    acoustic_cfg_json = json.load(open(args.acoustic_model.replace(".torch", ".json"), "r"))
    acoustic_cfg = AcousticConfig.from_json(acoustic_cfg_json)

    acoustic_model = CNN(acoustic_cfg)
    acoustic_model.float().to(device)
    try:
        acoustic_model.load_state_dict(torch.load(args.acoustic_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.acoustic_model, device))
        acoustic_model.load_state_dict(torch.load(args.acoustic_model, map_location=device))
    linguistic_cfg_json = json.load(open(args.linguistic_model.replace(".torch", ".json"), "r"))
    linguistic_cfg = LinguisticConfig.from_json(linguistic_cfg_json)

    linguistic_model = AttentionModel(linguistic_cfg)
    linguistic_model.float().to(device)

    try:
        linguistic_model.load_state_dict(torch.load(args.linguistic_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.linguistic_model, device))
        linguistic_model.load_state_dict(torch.load(args.linguistic_model, map_location=device))

    """Defining loss and optimizer"""
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = EnsembleModel(acoustic_model, linguistic_model)

    model_run_path = MODEL_PATH + "/" + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    model_weights_path = "{}/{}".format(model_run_path, "ensemble_model.torch")
    result_path = "{}/result.txt".format(model_run_path)
    os.makedirs(model_run_path, exist_ok=True)

    """Choosing hardware"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        print("Using GPU. Setting default tensor type to torch.cuda.FloatTensor")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("Using CPU. Setting default tensor type to torch.FloatTensor")
        torch.set_default_tensor_type("torch.FloatTensor")

    """Converting model to specified hardware and format"""
    model.float()
    model = model.to(device)

    """Defining loss and optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train_loss = 999
    best_val_loss = 999
    train_acc = 0
    epochs_without_improvement = 0

    """Running training"""
    for epoch in range(500):
        if epochs_without_improvement == 10:
            break

        val_loss, val_acc, val_weighted_acc, conf_mat = evaluate(model, val_iterator_acoustic, val_iterator_linguistic, criterion)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), model_weights_path)
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_weighted_acc = val_weighted_acc
            best_conf_mat = conf_mat
            epochs_without_improvement = 0
            log_success(
                " Epoch: {} | Val loss improved to {:.4f} | val acc: {:.3f} | weighted val acc: {:.3f} | train loss: {:.4f} | train acc: {:.3f} | saved model to {}.".format(
                    epoch, best_val_loss, best_val_acc, best_val_weighted_acc, train_loss, train_acc, model_weights_path
                ))

        train_loss, train_acc, train_weighted_acc, _ = train(model, train_iterator_acoustic, train_iterator_linguistic, optimizer, criterion, 0.0)

        epochs_without_improvement += 1

        if not epoch % 1:
            log(f'| Epoch: {epoch + 1} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}% '
                f'| Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.3f}%', True)

    model.load_state_dict(torch.load(model_weights_path))
    test_loss, test_acc, test_weighted_acc, conf_mat = evaluate(model, test_iterator_acoustic, test_iterator_linguistic, criterion)

    result = f'| Epoch: {epoch + 1} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Weighted Test Acc: {test_weighted_acc * 100:.2f}%\n Confusion matrix:\n {conf_mat}'
    log_major("Train acc: {}".format(train_acc))
    log_major(result)
    with open(result_path, "w") as file:
        file.write(result)