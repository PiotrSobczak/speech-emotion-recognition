import argparse
from torch.nn import NLLLoss
from os.path import isfile
import json
from matplotlib import pyplot as plt

from models import *
from train_utils import evaluate, eval_ensemble
from batch_iterator import BatchIterator
from data_loader import load_linguistic_dataset, load_spectrogram_dataset
from config import LinguisticConfig, AcousticSpectrogramConfig as AcousticConfig, EnsembleConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--linguistic_model", type=str, required=True)
    parser.add_argument("-a", "--acoustic_model", type=str, required=True)
    parser.add_argument("-e", "--ensemble_model", type=str, required=True)
    args = parser.parse_args()

    assert isfile(args.acoustic_model), "acoustic_model weights file does not exist"
    assert isfile(args.acoustic_model.replace(".torch", ".json")), "acoustic_model config file does not exist"
    assert isfile(args.linguistic_model), "linguistic_model weights file does not exist"
    assert isfile(args.linguistic_model.replace(".torch", ".json")), "linguistic_model config file does not exist"

    test_features_acoustic, test_labels_acoustic, val_features_acoustic, val_labels_acoustic, _, _ = load_spectrogram_dataset()
    test_features_linguistic, test_labels_linguistic, val_features_linguistic, val_labels_linguistic, _, _ = load_linguistic_dataset()

    test_iter_acoustic = BatchIterator(test_features_acoustic, test_labels_acoustic, 100)
    test_iter_linguistic = BatchIterator(test_features_linguistic, test_labels_linguistic, 100)
    val_iter_acoustic = BatchIterator(val_features_acoustic, val_labels_acoustic, 100)
    val_iter_linguistic = BatchIterator(val_features_linguistic, val_labels_linguistic, 100)

    assert np.array_equal(test_labels_acoustic, test_labels_linguistic), "Labels for acoustic and linguistic datasets are not the same!"

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
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(args.acoustic_model, device))
        acoustic_model.load_state_dict(torch.load(args.acoustic_model, map_location=device))
    linguistic_cfg_json = json.load(open(args.linguistic_model.replace(".torch", ".json"), "r"))
    linguistic_cfg = LinguisticConfig.from_json(linguistic_cfg_json)

    linguistic_model = AttentionModel(linguistic_cfg)
    linguistic_model.float().to(device)
    
    try:
        linguistic_model.load_state_dict(torch.load(args.linguistic_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(args.linguistic_model, device))
        linguistic_model.load_state_dict(torch.load(args.linguistic_model, map_location=device))

    """Defining loss and optimizer"""
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_loss, test_cm = evaluate(acoustic_model, test_iter_acoustic, criterion)
    print("Acoustic: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    test_loss, test_cm = evaluate(linguistic_model, test_iter_linguistic, criterion)
    print("Linguistic(asr=False): loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    ensemble_cfg_json = json.load(open(args.ensemble_model.replace(".torch", ".json"), "r"))
    ensemble_cfg = EnsembleConfig.from_json(ensemble_cfg_json)
    ensemble_model = EnsembleModel(ensemble_cfg)
    ensemble_model.float().to(device)

    try:
        ensemble_model.load_state_dict(torch.load(args.ensemble_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.ensemble_model, "cpu"))
        ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location="cpu"))

    ensemble_model.eval()

    test_loss, test_cm = eval_ensemble(
        ensemble_model, test_iter_acoustic, test_iter_linguistic, criterion
    )

    print("Featue-level ensemble: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    averageEnsemble = AverageEnsemble(acoustic_model, linguistic_model)

    test_loss, test_cm = eval_ensemble(
        averageEnsemble, test_iter_acoustic, test_iter_linguistic, NLLLoss().to(device)
    )
    print("Ensemble average: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    print("Searching for the optimal alpha on validation set...")

    alphas = {}
    for alpha in np.linspace(0.01, 0.99, 49):
        weightedAverageEnsemble = WeightedAverageEnsemble(acoustic_model, linguistic_model, alpha)
        _, val_cm = eval_ensemble(
            weightedAverageEnsemble, val_iter_acoustic, val_iter_linguistic, NLLLoss().to(device)
        )
        alphas[alpha] = val_cm.accuracy
    max_val = max(alphas.values())
    max_val_id = list(alphas.values()).index(max_val)
    max_val_alpha = list(alphas.keys())[max_val_id]
    assert alphas[max_val_alpha] == max_val

    fig = plt.figure()
    plt.plot(alphas.keys(), alphas.values())
    plt.xlabel("Alpha value")
    plt.ylabel("Weighted Acc on validation set")
    fig.savefig('alpha.png', dpi=fig.dpi)

    print("Found optimal alpha={}".format(max_val_alpha))

    weightedAverageEnsemble = WeightedAverageEnsemble(acoustic_model, linguistic_model, max_val_alpha)

    test_loss, test_cm = eval_ensemble(
        weightedAverageEnsemble, test_iter_acoustic, test_iter_linguistic, NLLLoss().to(device),
    )
    print("Ensemble weighted average: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    confidenceEnsemble = ConfidenceEnsemble(acoustic_model, linguistic_model)
    test_loss, test_cm = eval_ensemble(
        confidenceEnsemble, test_iter_acoustic, test_iter_linguistic, NLLLoss().to(device),
    )
    print("Ensemble confidence: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    print("-------------------------ASR---------------------------------------")
    test_features_linguistic, test_labels_linguistic, _, _, _, _ = load_linguistic_dataset(asr=True)
    test_iter_linguistic = BatchIterator(test_features_linguistic, test_labels_linguistic, 100)

    test_loss, test_cm = evaluate(linguistic_model, test_iter_linguistic, criterion)
    print("Linguistic(asr=True): loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    averageEnsemble = AverageEnsemble(acoustic_model, linguistic_model)
    test_loss, test_cm = eval_ensemble(
        averageEnsemble, test_iter_acoustic, test_iter_linguistic, NLLLoss().to(device),
    )
    print("Ensemble average: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    weightedAverageEnsemble = WeightedAverageEnsemble(acoustic_model, linguistic_model, max_val_alpha)
    test_loss, test_cm = eval_ensemble(
        weightedAverageEnsemble, test_iter_acoustic, test_iter_linguistic, NLLLoss().to(device),
    )
    print("Ensemble weighted average: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    confidenceEnsemble = ConfidenceEnsemble(acoustic_model, linguistic_model)
    test_loss, test_cm = eval_ensemble(
        confidenceEnsemble, test_iter_acoustic, test_iter_linguistic, NLLLoss().to(device),
    )
    print("Ensemble confidence: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}".format(test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))
