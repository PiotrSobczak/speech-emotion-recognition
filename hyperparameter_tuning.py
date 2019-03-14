import numpy as np

from train import run_training


NUM_ITERATIONS = 500

if __name__ == "__main__":
    run_training()
    for i in range(NUM_ITERATIONS):
        params = {}
        params["n_layers"] = np.random.randint(1, 2)
        params["hidden_dim"] = np.random.randint(256, 1000)
        params["dropout"] = 0.3 + np.random.rand() * 0.5
        params["reg_ratio"] = np.random.rand()*(10**(np.random.randint(-4, -2)))
        params["lr"] = np.random.rand() * (10 ** (np.random.randint(-2, 0)))
        params["batch_size"] = np.random.randint(32,96)
        params["seq_len"] = np.random.randint(20, 30)
        params["bidirectional"] = bool(np.random.randint(0,2))
        run_training(**params)
