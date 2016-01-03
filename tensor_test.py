import tensorflow as tf
import numpy as np


input_data = tf.placeholder(tf.int32, [20, 10])
embedding = tf.get_variable("embedding", [10000, 100])
inputs = tf.split(1, 10, tf.nn.embedding_lookup(embedding, input_data))