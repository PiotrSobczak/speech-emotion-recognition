import torch
import os
from time import gmtime, strftime
import json
import argparse
import numpy as np

from models import AttentionModel, CNN, EnsembleModel
from batch_iterator import BatchIterator
from data_loader import load_linguistic_dataset, load_spectrogram_dataset
from utils import log, log_major, log_success
from config import LinguisticConfig, AcousticSpectrogramConfig as AcousticConfig, EnsembleConfig
from train_utils import eval_ensemble, train_ensemble

MODEL_PATH = "saved_models"

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

    test_iter_acoustic = BatchIterator(test_features_acoustic, test_labels_acoustic, 100)
    test_iter_linguistic = BatchIterator(test_features_linguistic, test_labels_linguistic, 100)

    val_iter_acoustic = BatchIterator(val_features_acoustic, val_labels_acoustic, 100)
    val_iter_linguistic = BatchIterator(val_features_linguistic, val_labels_linguistic, 100)

    train_iter_acoustic = BatchIterator(train_features_acoustic, train_labels_acoustic, 100)
    train_iter_linguistic = BatchIterator(train_features_linguistic, train_labels_linguistic, 100)

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

    ensemble_cfg = EnsembleConfig(acoustic_cfg, linguistic_cfg)
    model = EnsembleModel(ensemble_cfg)

    model.load(acoustic_model, linguistic_model)
    tmp_run_path = MODEL_PATH + "/tmp_" + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    model_weights_path = "{}/{}".format(tmp_run_path, "ensemble_model.torch")
    model_config_path = "{}/{}".format(tmp_run_path, "ensemble_model.json")
    result_path = "{}/result.txt".format(tmp_run_path)
    os.makedirs(tmp_run_path, exist_ok=True)
    json.dump(ensemble_cfg.to_json(), open(model_config_path, "w"))

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
        if epochs_without_improvement == ensemble_cfg.patience:
            break

        val_loss, val_cm = eval_ensemble(model, val_iter_acoustic, val_iter_linguistic, criterion)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), model_weights_path)
            best_val_loss = val_loss
            best_val_acc = val_cm.accuracy
            best_val_unweighted_acc = val_cm.unweighted_accuracy
            best_conf_mat = val_cm
            epochs_without_improvement = 0
            log_success(
                " Epoch: {} | Val loss improved to {:.4f} | val acc: {:.3f} | weighted val acc: {:.3f} | train loss: {:.4f} | train acc: {:.3f} | saved model to {}.".format(
                    epoch, best_val_loss, best_val_acc, best_val_unweighted_acc, train_loss, train_acc, model_weights_path
                ))

        train_loss, train_cm = train_ensemble(model, train_iter_acoustic, train_iter_linguistic, optimizer, criterion, 0.0)
        train_acc = train_cm.accuracy

        epochs_without_improvement += 1

        if not epoch % 1:
            log(f'| Epoch: {epoch + 1} | Val Loss: {val_loss:.3f} | Val Acc: {val_cm.accuracy * 100:.2f}% '
                f'| Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.3f}%', True)

    model.load_state_dict(torch.load(model_weights_path))
    test_loss, test_cm = eval_ensemble(model, test_iter_acoustic, test_iter_linguistic, criterion)

    result = f'| Epoch: {epoch + 1} | Test Loss: {test_loss:.3f} | Test Acc: {test_cm.accuracy * 100:.2f}% | Weighted Test Acc: {test_cm.unweighted_accuracy * 100:.2f}%\n Confusion matrix:\n {test_cm}'
    log_major("Train acc: {}".format(train_acc))
    log_major(result)
    with open(result_path, "w") as file:
        file.write(result)
    output_path = "{}/ensemble_{:.3f}Acc_{:.3f}UAcc_{}".format(MODEL_PATH, test_cm.accuracy, test_cm.unweighted_accuracy, strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    os.rename(tmp_run_path, output_path)
