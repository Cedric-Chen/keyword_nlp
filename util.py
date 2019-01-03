import time

def record_cost(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		ret = func(*args, **kwargs)
		print(f"Function {func.__name__} takes {time.time() - start_time} seconds")
		return ret
	return wrapper