#%%
import time
import os
from itertools import islice
from multiprocessing import Pool
from functools import reduce, wraps
import operator

#%%
def record_cost(func):
	"""Function wrapper to output process time."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		ret = func(*args, **kwargs)
		print(f"Function '{func.__name__}' takes {time.time() - start_time} seconds")
		return ret
	return wrapper

#%%
class WordAnalyzer:
	""" Count word occurence in a file. """
	def __init__(self, filename):
		assert(os.path.exists(filename))
		self.filename = filename
		# self.file_opened = open(filename, 'r', encoding='utf-8')
	
	# def __del__(self):
	# 	self.file_opened.close()

	def _next_n_lines(self, file_opened, n):
		lines = list()
		for _ in range(n):
			line = file_opened.readline()
			if line:
				lines.append(line)
		return lines

	def _count_words(self, lines):
		count = dict()
		for line in lines:
			for word in line.strip().split(' '):
				if word not in count:
					count[word] = 1
				else:
					count[word] += 1
		return count

	def _merge_counts(self, counts1, counts2):
		for word, _ in counts2.items():
			if word not in counts1:
				counts1[word] = 0
			counts1[word] += counts2[word]
		return counts1

	@record_cost
	def count_words(self):
		with open(self.filename, 'r', encoding='utf-8') as f:
			results = list()
			line = f.readline()
			while line:
				results.append(self._count_words([line]))
				line = f.readline()

			counts = reduce(self._merge_counts, [p for p in results])
		print("len(counts)", len(counts))
		return counts
	
	#%%
	@record_cost
	def count_words_parallel(self, N_lines):
		with open(self.filename, 'r', encoding='utf-8') as f:
			with Pool(processes = 4) as pool:
				results = list()

				lines = self._next_n_lines(f, N_lines)
				while lines:
					results.append(pool.map_async(self._count_words, (lines,)))
					lines = self._next_n_lines(f, N_lines)	

				counts = reduce(self._merge_counts, [p.get()[0] for p in results])
		print("len(counts)", len(counts))
		return counts

	@record_cost
	def frequent_word_parallel(self, least_occurence, N_lines):
		counts = self.count_words_parallel(N_lines)
		sorted_counts = [x for x in counts.items() if x[1] >= least_occurence]
		sorted_counts = sorted(sorted_counts, key=operator.itemgetter(1), reverse=True)
		return sorted_counts

#%%
if __name__ == '__main__':
	filename = '/Users/puyangchen/go/src/github.com/agilab/cedric/words2.txt'
	assert(os.path.exists(filename))
	N_lines = 50000

	word_analyzer = WordAnalyzer(filename)
	print(len(word_analyzer.frequent_word_parallel(100, N_lines)))
	# for i, j in word_analyzer.frequent_word_parallel(100, N_lines):
	# 	print(i, j)