import time
from time import gmtime, strftime


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print ("{} - {} sec".format(method.__name__, te-ts))
        return result

    return timed


def log(log_message, verbose=False):
    if verbose:
        print("[{}]{}".format(strftime("%Y-%m-%d_%H:%M:%S", gmtime()), log_message))


def log_success(log_message):
    log("\033[32m{}\033[0m".format(log_message), True)


def log_major(log_message):
    log("\033[1m{}\033[0m".format(log_message), True)