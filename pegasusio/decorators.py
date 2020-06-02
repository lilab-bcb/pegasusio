import time
import gc
import functools


def timer(logger = None, precision: int = 2):
	""" Log the time spent for the function. """
	def inner_decorator(func):
		@functools.wraps(func)
		def wrapper_timer(*args, **kwargs):
			start = time.perf_counter()
			result = func(*args, **kwargs)
			end = time.perf_counter()
			message = f"Function '{func.__name__}' finished in {{:.{precision}f}}s.".format(end - start)

			if logger is None:
				print(message)
			else:
				logger.info(message)

			return result
		return wrapper_timer
	return inner_decorator


def run_gc(func):
	""" Run garbage collector """
	@functools.wraps(func)
	def wrapper_run_gc(*args, **kwargs):
		result = func(*args, **kwargs)
		gc.collect()
		return result
	return wrapper_run_gc
