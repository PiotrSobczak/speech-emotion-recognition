import numpy as np
import argparse

from train import run_training
from config import LinguisticConfig, AcousticConfig
from data_loader import load_acoustic_dataset, load_linguistic_dataset

NUM_ITERATIONS = 500

LINGUISTIC_TUNING = True

if __name__ == "__main__":
    for i in range(NUM_ITERATIONS):
        params = {}
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model_type", type=str, default="linguistic")
        args = parser.parse_args()

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
            run_training(LinguisticConfig(**params), test_features, test_labels, train_features, train_labels, val_features, val_labels)

        elif args.model_type == "acoustic":
            cfg = AcousticConfig()
            test_features, test_labels, val_features, val_labels, train_features, train_labels = load_acoustic_dataset()
            params["n_layers"] = np.random.randint(1, 4)
            params["hidden_dim"] = 50 #np.random.randint(10, 50)
            params["dropout"] = 0.0# 0.5 + np.random.rand() * 0.4
            params["dropout2"] = 0.5 + np.random.rand() * 0.45
            params["reg_ratio"] = 0.0 #np.random.rand()*0.0015
            params["lr"] = 0.001 #np.random.rand() * (10 ** (np.random.randint(-2, 0)))
            params["batch_size"] = 96 #np.random.randint(26,256)
            #params["seq_len"] = np.random.randint(20, 30)
            params["bidirectional"] = bool(np.random.randint(0,2))
            run_training(AcousticConfig(**params), test_features, test_labels, train_features, train_labels, val_features, val_labels)

        else:
            raise Exception("model_type parameter has to be one of [acoustic|linguistic]")
