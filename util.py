#%%
import time
from datetime import datetime
from functools import wraps

#%%
def record_cost(func):
	"""Function wrapper to output process time."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		print(datetime.now(), f"Function '{func.__name__}' starts.")
		ret = func(*args, **kwargs)
		print(datetime.now(), "Function '{}' takes {:.2f} seconds".format(func.__name__, time.time()-start_time))
		return ret
	return wrapper
