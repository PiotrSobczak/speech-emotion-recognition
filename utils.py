from time import gmtime, strftime, time
import torch


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print ("{} - {} sec".format(method.__name__, te-ts))
        return result

    return timed


def get_datetime():
    return strftime("%Y-%m-%d_%H:%M:%S", gmtime())


def log(log_message, verbose=False):
    if verbose:
        print("[{}]{}".format(get_datetime(), log_message))


def log_success(log_message, verbose=True):
    log("\033[32m{}\033[0m".format(log_message), verbose)


def log_major(log_message, verbose=True):
    log("\033[1m{}\033[0m".format(log_message), verbose)


def set_default_tensor():
    if torch.cuda.is_available():
        print("Using GPU. Setting default tensor type to torch.cuda.FloatTensor")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("Using CPU. Setting default tensor type to torch.FloatTensor")
        torch.set_default_tensor_type("torch.FloatTensor")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
