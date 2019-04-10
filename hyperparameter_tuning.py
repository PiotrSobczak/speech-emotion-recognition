import numpy as np

from train import run_training
from config import LinguisticConfig

NUM_ITERATIONS = 500

if __name__ == "__main__":
    run_training(LinguisticConfig)
    for i in range(NUM_ITERATIONS):
        params = {}
        #params["n_layers"] = np.random.randint(1, 2)
        #params["hidden_dim"] = np.random.randint(256, 1000)
        params["dropout"] = 0.5 + np.random.rand() * 0.4
        params["dropout2"] = 0.2 + np.random.rand() * 0.6
        #params["reg_ratio"] = np.random.rand()*0.0015
        #params["lr"] = 0.003 #np.random.rand() * (10 ** (np.random.randint(-2, 0)))
        params["batch_size"] = np.random.randint(64,256)
        params["seq_len"] = np.random.randint(20, 30)
        #params["bidirectional"] = bool(np.random.randint(0,2))
        run_training(LinguisticConfig(**params))

        run_training(LinguisticConfig(**params))
