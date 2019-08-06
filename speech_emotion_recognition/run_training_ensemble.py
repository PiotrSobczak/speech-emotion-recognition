import torch
import os
import json
import argparse

from speech_emotion_recognition.models import AttentionLSTM, CNN, FeatureEnsemble
from speech_emotion_recognition.batch_iterator import EnsembleBatchIterator
from speech_emotion_recognition.data_loader import load_linguistic_dataset, load_spectrogram_dataset, create_batches
from speech_emotion_recognition.config import LinguisticConfig, AcousticSpectrogramConfig, EnsembleConfig
from speech_emotion_recognition.train import train

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

    test_iterator_ac, train_iterator_ac, validation_iterator_ac = create_batches(*load_spectrogram_dataset(), 100)
    test_iterator_li, train_iterator_li, validation_iterator_li = create_batches(*load_linguistic_dataset(), 100)
    test_iterator = EnsembleBatchIterator(test_iterator_ac, test_iterator_li)
    train_iterator = EnsembleBatchIterator(train_iterator_ac, train_iterator_li)
    validation_iterator = EnsembleBatchIterator(validation_iterator_ac, validation_iterator_li)

    """Converting model to specified hardware and format"""
    acoustic_cfg_json = json.load(open(args.acoustic_model.replace(".torch", ".json"), "r"))
    acoustic_cfg = AcousticSpectrogramConfig(**acoustic_cfg_json)
    acoustic_model = CNN(acoustic_cfg)
    acoustic_model.load(args.acoustic_model)

    linguistic_cfg_json = json.load(open(args.linguistic_model.replace(".torch", ".json"), "r"))
    linguistic_cfg = LinguisticConfig(**linguistic_cfg_json)
    linguistic_model = AttentionLSTM(linguistic_cfg)
    linguistic_model.load(args.linguistic_model)

    """Defining loss and optimizer"""
    criterion = torch.nn.CrossEntropyLoss()

    ensemble_cfg = EnsembleConfig(acoustic_cfg, linguistic_cfg)
    model = FeatureEnsemble(ensemble_cfg, acoustic_model, linguistic_model)

    train(model, ensemble_cfg, test_iterator, train_iterator, validation_iterator)


