#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import os
import socket
import collections
import math
import time

#%%
mp_dir = "/Users/puyangchen/go/src/github.com/agilab/cedric/tensorflow"
deepbox_dir = "/home/chenpuyang/Projects/keyword/keyword_nlp"

vocabulary_size = 1000000
filename = 'words.txt'
num_steps = 1000000

if 'macbook' in socket.gethostname().lower():
    assert(os.path.exists(mp_dir))
    os.chdir(mp_dir)
    curdir = mp_dir
    log_dir = os.path.abspath(os.path.join(curdir, '..', 'log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    input_file = os.path.abspath(os.path.join(curdir, '..', filename))
    assert os.path.exists(input_file)
elif 'deepbox' in socket.gethostname().lower():
    assert(os.path.exists(deepbox_dir))
    os.chdir(deepbox_dir)
    curdir = deepbox_dir
    log_dir = os.path.abspath(os.path.join(curdir, '..', 'log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    input_file = os.path.abspath(os.path.join(curdir, '..', 'data', filename))
    assert os.path.exists(input_file)
else:
    print('Unknown host.')

#%%
from util import record_cost

#%%
window_size = 2
batch_size = 128
embedding_size = 200

num_sampled = 64
learning_rate = 1.0

#%%
@record_cost
def read_file(filename):
    with open(filename, "r", encoding='UTF-8') as f:
        file = []
        line = f.readline()
        while line:
            file.append(line.strip().split(" "))
            line = f.readline()
        return file

words = read_file(input_file)

#%%
@record_cost
def build_dataset(words, n_words):
    index = dict()
    inverted_index = dict()
    raw_data = list()
    count = [["UNK", -1]]

    count.extend(collections.Counter([y for x in words for y in x]).most_common(n_words))
    for word, _ in count:
        n = len(index)-1
        index[word] = n
        inverted_index[n] = word

    unk_count = 0
    for i in words:
        ids = list()
        for j in i:
            id = index.get(j, -1)
            ids.append(id)
            if id == -1:
                unk_count += 1
        raw_data.append(ids)
    count[0][1] = unk_count
    
    return index, inverted_index, raw_data, count

index, inverted_index, raw_data, count = build_dataset(words, vocabulary_size)
del words

#%%
line_index = 0
element_index = 0
window_index = -window_size
# @record_cost
def generate_input(raw_data, window_size, batch_size):
    global line_index, element_index, window_index

    word = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    iid = 0

    while line_index < len(raw_data):
        line = raw_data[line_index]
        while element_index < len(line):
            element = line[element_index]

            # positive cases
            if element == -1:
                element_index += 1
                window_index += 1
                continue

            while window_index <= window_size:
                # print('line', line_index, 'element', element_index, 'window', window_index)
                if element_index+window_index<0 or element_index+window_index>=len(line) or window_index==0:
                    window_index += 1
                    continue
                if line[element_index+window_index] == -1:
                    window_index += 1
                    continue

                # print('word', element, 'label', line[element_index+window_index])
                word[iid] = element
                label[iid][0] = line[element_index+window_index]

                window_index += 1
                iid += 1

                if iid == batch_size:
                    return word, label
            
            element_index += 1
            window_index = -window_size
        
        line_index += 1
        element_index = 0
        if line_index == len(raw_data):
            line_index = 0

# #%%
# word, label = generate_input(raw_data, window_size, 10)
# print(word)
# print(label)

#%%
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)
                )
            )
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size
            )
        )
    
    tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings/norm

    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

#%%
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    if os.path.exists(os.path.join(log_dir, 'model.ckpt.index')):
        saver.restore(sess, os.path.join(log_dir, 'model.ckpt'))
        print('restore from checkpoint file')
    else:
        init.run()
        print('initialized')

    average_loss = 0
    last_checkpoint_time = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_input(raw_data, window_size, batch_size)
        feed_dict = {
            train_inputs: batch_inputs,
            train_labels: batch_labels,
        }
    
        run_metadata = tf.RunMetadata()

        _, summary, loss_val = sess.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata
        )
        average_loss += loss_val

        writer.add_summary(summary, step)
        if step == (num_steps-1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
        
        now = time.time()
        if now - last_checkpoint_time > 20 * 1:
            saver.save(sess, os.path.join(log_dir, 'model.ckpt'))
            last_checkpoint_time = time.time()

    final_embeddings = normalized_embeddings.eval()
    with open(log_dir+'/metadata.tsv', 'w', encoding='utf-8') as f:
        for i in range(vocabulary_size):
            out = ['{:.6f}'.format(a) for a in final_embeddings[i]]
            f.write(inverted_index[i]+' '+' '.join(out)+'\n')
    

    # Create a configuration for visualizeing embeddings with the labels in TensorBoard
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
