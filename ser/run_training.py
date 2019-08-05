import argparse

from ser.models import AttentionLSTM as RNN, CNN
from ser.batch_iterator import BatchIterator
from ser.data_loader import load_linguistic_dataset, load_acoustic_features_dataset, load_spectrogram_dataset
from ser.utils import set_default_tensor
from ser.config import LinguisticConfig, AcousticLLDConfig, AcousticSpectrogramConfig
from ser.train import train

MODEL_PATH = "saved_models"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="linguistic")
    args = parser.parse_args()
    set_default_tensor()

    if args.model_type == "linguistic":
        cfg = LinguisticConfig()
        test_features, test_labels, val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
        model = RNN(cfg)
    elif args.model_type == "acoustic-lld":
        cfg = AcousticLLDConfig()
        test_features, test_labels, val_features, val_labels, train_features, train_labels = load_acoustic_features_dataset()
        model = RNN(cfg)
    elif args.model_type == "acoustic-spectrogram":
        cfg = AcousticSpectrogramConfig()
        test_features, test_labels, val_features, val_labels, train_features, train_labels = load_spectrogram_dataset()
        model = CNN(cfg)

    else:
        raise Exception("model_type parameter has to be one of [linguistic|acoustic-lld|acoustic-spectrogram]")

    """Creating data generators"""
    test_iterator = BatchIterator(test_features, test_labels)
    train_iterator = BatchIterator(train_features, train_labels, cfg.batch_size)
    validation_iterator = BatchIterator(val_features, val_labels)

    """Running training"""
    train(model, cfg, test_iterator, train_iterator, validation_iterator)
