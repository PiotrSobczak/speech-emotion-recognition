import numpy as np
import argparse
import torch
from os.path import isfile
import json

from models import AttentionModel
from train_utils import evaluate, evaluate_ensemble
from batch_iterator import BatchIterator
from data_loader import load_linguistic_dataset, load_acoustic_dataset
from config import LinguisticConfig, AcousticConfig

MODEL_PATH = "saved_models"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--linguistic_model", type=str, required=True)
    parser.add_argument("-a", "--acoustic_model", type=str, required=True)
    args = parser.parse_args()

    assert isfile(args.acoustic_model), "acoustic_model weights file does not exist"
    assert isfile(args.acoustic_model.replace(".torch", ".json")), "acoustic_model config file does not exist"
    assert isfile(args.linguistic_model), "linguistic_model weights file does not exist"
    assert isfile(args.linguistic_model.replace(".torch", ".json")), "linguistic_model config file does not exist"

    val_features_acoustic, val_labels_acoustic, _, _ = load_acoustic_dataset()
    val_iterator_acoustic = BatchIterator(val_features_acoustic, val_labels_acoustic, 100)
    val_features_linguistic, val_labels_linguistic, _, _ = load_linguistic_dataset()
    val_iterator_linguistic = BatchIterator(val_features_linguistic, val_labels_linguistic, 100)

    assert np.array_equal(val_labels_acoustic, val_labels_linguistic), "Labels for acoustic and linguistic datasets are not the same!"

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

    acoustic_model = AttentionModel(acoustic_cfg)
    acoustic_model.float().to(device)
    acoustic_model.load_state_dict(torch.load(args.acoustic_model, map_location=device))

    linguistic_cfg_json = json.load(open(args.linguistic_model.replace(".torch", ".json"), "r"))
    linguistic_cfg = LinguisticConfig.from_json(linguistic_cfg_json)

    linguistic_model = AttentionModel(linguistic_cfg)
    linguistic_model.float().to(device)
    linguistic_model.load_state_dict(torch.load(args.linguistic_model, map_location=device))

    """Defining loss and optimizer"""
    criterion = torch.nn.CrossEntropyLoss().to(device)

    val_loss, val_acc, val_weighted_acc, conf_mat = evaluate(acoustic_model, val_iterator_acoustic, criterion)
    print("Acoustic: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(val_loss, val_acc, val_weighted_acc, conf_mat))

    val_loss, val_acc, val_weighted_acc, conf_mat = evaluate(linguistic_model, val_iterator_linguistic, criterion)
    print("Linguistic: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(val_loss, val_acc, val_weighted_acc, conf_mat))

    val_loss, val_acc, val_weighted_acc, conf_mat = evaluate_ensemble(acoustic_model, linguistic_model, val_iterator_acoustic, val_iterator_linguistic, torch.nn.NLLLoss().to(device))
    print("Ensemble: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(val_loss, val_acc, val_weighted_acc, conf_mat))
