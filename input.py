#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from multiprocessing import Pool
from functools import reduce, wraps
from datetime import datetime
import operator
import os
import socket
import multiprocessing
from collections import Counter

import numpy as np
import tensorflow as tf

#%%
mp_dir = "/Users/puyangchen/go/src/github.com/agilab/cedric/tensorflow"
deepbox_dir = "/home/chenpuyang/Projects/keyword/keyword_nlp"
infile_name = 'words2.txt'
outfile_name = 'mod_data.csv'

if 'macbook' in socket.gethostname().lower():
    assert(os.path.exists(mp_dir))
    os.chdir(mp_dir)
    curdir = mp_dir
    input_file = os.path.abspath(os.path.join(curdir, '..', infile_name))
    output_file = os.path.abspath(os.path.join(curdir, '..', outfile_name))
    assert os.path.exists(input_file)
elif 'deepbox' in socket.gethostname().lower():
    assert(os.path.exists(deepbox_dir))
    os.chdir(deepbox_dir)
    curdir = deepbox_dir
    input_file = os.path.abspath(os.path.join(curdir, '..', 'data', infile_name))
    output_file = os.path.abspath(os.path.join(curdir, '..', 'data', outfile_name))
    assert os.path.exists(input_file)
else:
    print('Unknown host.')

from util import record_cost

#%%
class SampleWord:
	def __init__(self, word_counts):
		self.word_counts = word_counts

		self.words = [x[0] for x in self.word_counts]
		self.counts = [x[1] for x in self.word_counts]
		self.probs = [x/sum(self.counts) for x in self.counts]

	# @record_cost
	def sample(self):
		return np.random.choice(self.words, size=1, p=self.probs)[0]

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
	def __init__(self, input_file, output_file, window_size, num_negative_samples, sample_func):
		self.input_file = input_file 
		self.output_file = output_file
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
				labels.append(1)
			# negative samples
			for _ in range(self.num_negative_samples):
				features.append((words[i], self.sample_func()))
				labels.append(0)
				
		return features, labels


	def _parse_lines(self, lines):
		features = list()
		labels = list()
		for line in lines:
			feature, label = self._parse_line(line)
			features.extend(feature)
			labels.extend(label)
		return features, labels

	# def _merge_data(self, data1, data2):
	# 	data1[0].extend(data2[0])
	# 	data1[1].extend(data2[1])
	# 	return data1[0], data1[1]

	@record_cost
	def generate_dataset(self, N_lines=50000):
		with open(self.input_file, 'r', encoding='utf-8') as inf:
			with open(self.output_file, 'w', encoding='utf-8') as of:
				with Pool(processes = multiprocessing.cpu_count()) as pool:
					results = list()

					lines = next_n_lines(inf, N_lines)
					while lines:
						results.append(pool.map_async(self._parse_lines, (lines,)))
						lines = next_n_lines(inf, N_lines)	

					num_of_process = len(results)
					print("number of process", num_of_process)

					for count, p in enumerate(results):
						a, b = p.get()[0]
						for i, _ in enumerate(a):
							of.write(a[i][0] + ',' + a[i][1] + ',' + str(b[i]) + '\n')
						print(datetime.now(), 'process', count, 'finished')

#%%
sample_func = SampleWord(WordAnalyzer(input_file).most_common_parallel()).sample
window_size = 2
num_negative_samples = 2

#%%
input = Input(input_file, output_file, window_size, num_negative_samples, sample_func)
input.generate_dataset()
