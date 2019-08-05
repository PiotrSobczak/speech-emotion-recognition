import numpy as np
import argparse

from ser.train import train
from ser.config import LinguisticConfig, AcousticSpectrogramConfig, AcousticLLDConfig
from ser.data_loader import load_acoustic_features_dataset, load_linguistic_dataset, load_spectrogram_dataset
from ser.models import AttentionLSTM as RNN, CNN
from ser.utils import set_default_tensor
from ser.batch_iterator import BatchIterator

NUM_ITERATIONS = 500

LINGUISTIC_TUNING = True

if __name__ == "__main__":
    for i in range(NUM_ITERATIONS):
        params = {}
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model_type", type=str, default="linguistic")
        args = parser.parse_args()

        set_default_tensor()

        if args.model_type == "linguistic":
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
            params["hidden_dim"] = np.random.randint(50, 500)
            params["dropout"] = 0.5 + np.random.rand() * 0.4
            params["dropout2"] = 0.2 + np.random.rand() * 0.6
            params["reg_ratio"] = np.random.rand()*0.0015
            params["batch_size"] = np.random.randint(64, 256)
            params["seq_len"] = np.random.randint(20, 30)
            cfg = LinguisticConfig(**params)
            model = RNN(cfg)

        elif args.model_type == "acoustic-lld":
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_acoustic_features_dataset()
            params["n_layers"] = np.random.randint(1, 4)
            params["hidden_dim"] = np.random.randint(10, 100)
            params["dropout"] = 0.5 + np.random.rand() * 0.4
            params["dropout2"] = 0.5 + np.random.rand() * 0.45
            params["reg_ratio"] = np.random.rand()*0.0015
            params["batch_size"] = np.random.randint(26,256)
            params["bidirectional"] = bool(np.random.randint(0, 2))
            cfg = AcousticLLDConfig(**params)
            model = RNN(cfg)

        elif args.model_type == "acoustic-spectrogram":
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_spectrogram_dataset()
            params["fc_size"] = np.random.randint(10, 200)
            params["dropout"] = 0.3 + np.random.rand() * 0.6
            cfg = AcousticSpectrogramConfig(**params)
            model = CNN(cfg)

        else:
            raise Exception("model_type parameter has to be one of [linguistic|acoustic-lld|acoustic-spectrogram]")

        print("Subsets sizes: test_features:{}, test_labels:{}, val_features:{}, val_labels:{}, train_features:{}, train_labels:{}".format(
            test_features.shape[0], test_labels.shape[0], val_features.shape[0], val_labels.shape[0], train_features.shape[0], train_labels.shape[0])
        )

        """Creating data generators"""
        test_iterator = BatchIterator(test_features, test_labels)
        train_iterator = BatchIterator(train_features, train_labels, cfg.batch_size)
        validation_iterator = BatchIterator(val_features, val_labels)

        train(model, cfg, test_iterator, train_iterator, validation_iterator)
