import tensorflow as tf

'''
constant system parameters
'''
inputDim = 300
hiddenDim = 150
outputDim = 5

'''
shared model parameters
'''
with tf.variable_scope("composor") as composor_scope:
	iw = tf.get_variable("hidden_weight", [hiddenDim, inputDim])
	ib = tf.get_variable("hidden_bias", [hiddenDim])
	hw = tf.get_variable("hidden_weight", [hiddenDim, hiddenDim])
	hb = tf.get_variable("hidden_bias", [hiddenDim])
	ow = tf.get_variable("output_weight", [outputDim, hiddenDim])
	ob = tf.get_variable("output_bias", [outputDim])