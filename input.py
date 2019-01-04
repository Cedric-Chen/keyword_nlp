#%%
import os
os.chdir('/Users/puyangchen/go/src/github.com/agilab/cedric/tensorflow')

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# from itertools import product
from multiprocessing import Pool
from functools import reduce, wraps
import operator
import multiprocessing
from collections import Counter

import numpy as np
import tensorflow as tf

from util import record_cost


#%%
class SampleWord:
	def __init__(self, word_counts):
		self.word_counts = word_counts

	def sample(self):
		words = [x[0] for x in self.word_counts]
		counts = [x[1] for x in self.word_counts]
		probs = [x/sum(counts) for x in counts]

		return np.random.choice(words, size=1, p=probs)[0]

def next_n_lines(file_opened, n):
	lines = list()
	for _ in range(n):
		line = file_opened.readline()
		if line:
			lines.append(line)
	return lines

#%%
class WordAnalyzer:
	""" Count word occurence in a file. """
	def __init__(self, filename):
		assert(os.path.exists(filename))
		self.filename = filename
		# self.file_opened = open(filename, 'r', encoding='utf-8')

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
			with Pool(processes = multiprocessing.cpu_count()) as pool:
				results = list()

				lines = next_n_lines(f, N_lines)
				while lines:
					results.append(pool.map_async(self._count_words, (lines,)))
					lines = next_n_lines(f, N_lines)	

				counts = reduce(self._merge_counts, [p.get()[0] for p in results])
		print("number of words", len(counts))
		return counts

	def frequent_word_parallel(self, least_occurence=100, N_lines=50000):
		counts = self.count_words_parallel(N_lines)
		sorted_counts = [x for x in counts.items() if x[1] >= least_occurence]
		sorted_counts = sorted(sorted_counts, key=operator.itemgetter(1), reverse=True)
		return sorted_counts

	def most_common_parallel(self, N=5000, N_lines=50000):
		counts = self.count_words_parallel(N_lines)
		return Counter(counts).most_common(N)

class Input:
	def __init__(self, filename, window_size, num_negative_samples, sample_func):
		self.filename = filename
		self.window_size = window_size
		self.num_negative_samples = num_negative_samples
		self.sample_func = sample_func

	def _parse_line(self, line):
		features = list()
		labels = list()

		words = line.strip().split(' ')
		# print(words)
		for i, _ in enumerate(words):
			# positive samples
			for j in range(-self.window_size, self.window_size+1):
				if j == 0 or i+j < 0 or i+j >= len(words):
					continue
				features.append((words[i], words[i+j]))
				labels.append(True)
			# negative samples
			for k in range(self.num_negative_samples):
				features.append((words[i], self.sample_func()))
				labels.append(False)
				
		return features, labels


	def _parse_lines(self, lines):
		features = list()
		labels = list()
		for line in lines:
			feature, label = self._parse_line(line)
			features.extend(feature)
			labels.extend(label)
		return features, labels

	def _merge_data(self, data1, data2):
		data1[0].extend(data2[0])
		data1[1].extend(data2[1])
		return data1[0], data1[1]

	@record_cost
	def generate_dataset(self, filename, N_lines):
		with open(self.filename, 'r', encoding='utf-8') as f:
			with Pool(processes = multiprocessing.cpu_count()) as pool:
				results = list()

				lines = next_n_lines(f, N_lines)
				while lines:
					results.append(pool.map_async(self._parse_lines, (lines,)))
					lines = next_n_lines(f, N_lines)	

				data = reduce(self._merge_data, [p.get()[0] for p in results])

		return data[0], data[1] 

#%%
filename = '/Users/puyangchen/go/src/github.com/agilab/cedric/words1.txt'

sample_func = SampleWord(WordAnalyzer(filename).frequent_word_parallel()).sample
window_size = 2
num_negative_samples = 2

#%%
input = Input(filename, window_size, num_negative_samples, sample_func)
a, b = input.generate_dataset(filename, 2)
assert len(a) == len(b)
print("data length", len(a))
for i in range(10):
	print(a[i], b[i])