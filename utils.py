import time


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
        print(log_message)


def log_success(log_message):
        print("\033[32m{}\033[0m".format(log_message))


def log_major(log_message):
    print("\033[1m{}\033[0m".format(log_message))