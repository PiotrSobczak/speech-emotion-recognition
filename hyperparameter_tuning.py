import numpy as np
import argparse

from train import run_training
from config import LinguisticConfig, AcousticSpectrogramConfig, AcousticLLDConfig
from data_loader import load_acoustic_features_dataset, load_linguistic_dataset, load_spectrogram_dataset
from models import AttentionModel as RNN, CNN
from utils import get_device, set_default_tensor

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
            #params["n_layers"] = np.random.randint(1, 2)
            #params["hidden_dim"] = np.random.randint(256, 1000)
            params["dropout"] = 0.5 + np.random.rand() * 0.4
            params["dropout2"] = 0.2 + np.random.rand() * 0.6
            #params["reg_ratio"] = np.random.rand()*0.0015
            #params["lr"] = 0.003 #np.random.rand() * (10 ** (np.random.randint(-2, 0)))
            params["batch_size"] = np.random.randint(64,256)
            params["seq_len"] = np.random.randint(20, 30)
            #params["bidirectional"] = bool(np.random.randint(0,2))
            cfg = LinguisticConfig(**params)
            model = RNN(cfg)

        elif args.model_type == "acoustic-lld":
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_acoustic_features_dataset()
            params["n_layers"] = np.random.randint(1, 4)
            params["hidden_dim"] = 50 #np.random.randint(10, 50)
            params["dropout"] = 0.0# 0.5 + np.random.rand() * 0.4
            params["dropout2"] = 0.5 + np.random.rand() * 0.45
            params["reg_ratio"] = 0.0 #np.random.rand()*0.0015
            params["lr"] = 0.001 #np.random.rand() * (10 ** (np.random.randint(-2, 0)))
            params["batch_size"] = 96 #np.random.randint(26,256)
            #params["seq_len"] = np.random.randint(20, 30)
            params["bidirectional"] = bool(np.random.randint(0,2))
            cfg = AcousticLLDConfig(**params)
            model = RNN(cfg)

        elif args.model_type == "acoustic-spectrogram":
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_spectrogram_dataset()
            #params["fc_size"] = np.random.randint(10, 200)
            params["dropout"] = 0.3 + np.random.rand() * 0.6
            cfg = AcousticSpectrogramConfig(**params)
            model = CNN(cfg)

        else:
            raise Exception("model_type parameter has to be one of [linguistic|acoustic-lld|acoustic-spectrogram]")

        print("Subsets sizes: test_features:{}, test_labels:{}, val_features:{}, val_labels:{}, train_features:{}, train_labels:{}".format(
            test_features.shape[0], test_labels.shape[0], val_features.shape[0], val_labels.shape[0], train_features.shape[0], train_labels.shape[0])
        )

        """Converting model to specified hardware and format"""
        model.float()
        model = model.to(get_device())

        run_training(model, cfg, test_features, test_labels, train_features, train_labels, val_features, val_labels)
