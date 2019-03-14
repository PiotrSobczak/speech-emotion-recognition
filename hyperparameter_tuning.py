import numpy as np

from train import run_training


NUM_ITERATIONS = 200

if __name__ == "__main__":
    for i in range(NUM_ITERATIONS):
        params = {}
        params["n_layers"] = np.random.randint(1, 4)
        params["hidden_dim"] = np.random.randint(64, 1200)
        params["dropout"] = 0.1 + np.random.rand() * 0.85
        params["reg_ratio"] = np.random.rand()*(10**(np.random.randint(-4, -2)))
        params["batch_size"] = np.random.randint(16,150)
        params["seq_len"] = np.random.randint(10, 50)
        run_training(**params)