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
	initializer = tf.random_uniform_initializer(-0.05, 0.05)
	iw = tf.get_variable("hidden_weight", [hiddenDim, inputDim], initializer)
	ib = tf.get_variable("hidden_bias", [hiddenDim], initializer)
	hw = tf.get_variable("hidden_weight", [hiddenDim, hiddenDim], initializer)
	hb = tf.get_variable("hidden_bias", [hiddenDim], initializer)
	ow = tf.get_variable("output_weight", [outputDim, hiddenDim], initializer)
	ob = tf.get_variable("output_bias", [outputDim], initializer)