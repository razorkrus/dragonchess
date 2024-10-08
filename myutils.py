import functools
import time


def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        name = func.__name__
        print(f"{name} execution time: {execution_time} seconds")
        return result

    return wrapper
