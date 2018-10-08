from functools import wraps
import time


def for_all_methods(decorator, exceptions=None):
    if exceptions is None:
        exceptions = []

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr), exceptions))
        return cls
    return decorate


def log(f, exceptions):
    wraps(f)

    def wrapped(*args, **kwargs):
        t0 = time.time()
        r = f(*args, **kwargs)
        if f.__name__ not in exceptions:
            print('[LOG] ' + f.__name__ + ' exited after {}s'.format(round(time.time() - t0), 2))
        return r
    return wrapped
