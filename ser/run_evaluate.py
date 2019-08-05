import argparse
import json

from torch.nn import CrossEntropyLoss
from os.path import isfile

from ser.models import *
from ser.model_utils import run_epoch_eval, search_for_optimal_alpha
from ser.batch_iterator import BatchIterator, EnsembleBatchIterator
from ser.data_loader import load_linguistic_dataset, load_spectrogram_dataset
from ser.config import LinguisticConfig, AcousticSpectrogramConfig as AcousticConfig, EnsembleConfig


SCORE_STR = "{}: loss: {}, acc: {}. unweighted acc: {}, conf_mat: \n{}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--linguistic_model", type=str, required=True)
    parser.add_argument("-a", "--acoustic_model", type=str, required=True)
    parser.add_argument("-e", "--feature_ensemble", type=str, required=True)
    args = parser.parse_args()

    acoustic_config = args.acoustic_model.replace(".torch", ".json")
    linguistic_config = args.linguistic_model.replace(".torch", ".json")
    assert isfile(args.acoustic_model), "{} does not exist".format(args.acoustic_model)
    assert isfile(acoustic_config), "{} config file does not exist".format(acoustic_config)
    assert isfile(args.linguistic_model), "{} does not exist".format(args.linguistic_model)
    assert isfile(linguistic_config), "{} file does not exist".format(linguistic_config)

    test_features_acoustic, test_labels_acoustic, val_features_acoustic, val_labels_acoustic, _, _ = load_spectrogram_dataset()
    test_features_linguistic, test_labels_linguistic, val_features_linguistic, val_labels_linguistic, _, _ = load_linguistic_dataset()

    test_iter_acoustic = BatchIterator(test_features_acoustic, test_labels_acoustic, 100)
    val_iter_acoustic = BatchIterator(val_features_acoustic, val_labels_acoustic, 100)
    test_iter_linguistic = BatchIterator(test_features_linguistic, test_labels_linguistic, 100)
    val_iter_linguistic = BatchIterator(val_features_linguistic, val_labels_linguistic, 100)
    test_iter_ensemble = EnsembleBatchIterator(test_iter_acoustic, test_iter_linguistic, 100)
    val_iter_ensemble = EnsembleBatchIterator(val_iter_acoustic, val_iter_linguistic, 100)

    assert np.array_equal(test_labels_acoustic, test_labels_linguistic), "Inconsistent acoustic and linguistic Labels!"

    """Loading models"""
    acoustic_cfg_json = json.load(open(args.acoustic_model.replace(".torch", ".json"), "r"))
    acoustic_cfg = AcousticConfig(**acoustic_cfg_json)
    acoustic_model = CNN(acoustic_cfg)
    acoustic_model.load(args.acoustic_model)

    linguistic_cfg_json = json.load(open(args.linguistic_model.replace(".torch", ".json"), "r"))
    linguistic_cfg = LinguisticConfig(**linguistic_cfg_json)
    linguistic_model = AttentionLSTM(linguistic_cfg)
    linguistic_model.load(args.linguistic_model)
    
    ensemble_cfg_json = json.load(open(args.feature_ensemble.replace(".torch", ".json"), "r"))
    ensemble_cfg = EnsembleConfig.from_json(ensemble_cfg_json)
    feature_ensemble = FeatureEnsemble(ensemble_cfg)
    feature_ensemble.load(args.feature_ensemble)
    feature_ensemble.eval()

    optimal_alpha = search_for_optimal_alpha(acoustic_model, linguistic_model, val_iter_ensemble)
    weightedAverageEnsemble = WeightedAverageEnsemble(acoustic_model, linguistic_model, optimal_alpha)
    averageEnsemble = AverageEnsemble(acoustic_model, linguistic_model)
    confidenceEnsemble = ConfidenceEnsemble(acoustic_model, linguistic_model)

    """Evaluating models"""
    loss_function = CrossEntropyLoss()
    test_loss, test_cm = run_epoch_eval(acoustic_model, test_iter_acoustic, loss_function)
    print(SCORE_STR.format("Acoustic", test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    test_loss, test_cm = run_epoch_eval(linguistic_model, test_iter_linguistic, loss_function)
    print(SCORE_STR.format("Linguistic", test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    for ensemble_model in [feature_ensemble, weightedAverageEnsemble, averageEnsemble, confidenceEnsemble]:
        test_loss, test_cm = run_epoch_eval(ensemble_model, test_iter_ensemble, loss_function)
        print(SCORE_STR.format(ensemble_model.name, test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    test_features_linguistic, test_labels_linguistic, _, _, _, _ = load_linguistic_dataset(asr=True)
    test_iter_linguistic = BatchIterator(test_features_linguistic, test_labels_linguistic)

    test_loss, test_cm = run_epoch_eval(linguistic_model, test_iter_linguistic, loss_function)
    print(SCORE_STR.format("Linguistic(ASR)", test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))

    for ensemble_model in [weightedAverageEnsemble, averageEnsemble, confidenceEnsemble, feature_ensemble]:
        test_loss, test_cm = run_epoch_eval(ensemble_model, test_iter_ensemble, loss_function)
        print(SCORE_STR.format(ensemble_model.name+"(ASR)", test_loss, test_cm.accuracy, test_cm.unweighted_accuracy, test_cm))
