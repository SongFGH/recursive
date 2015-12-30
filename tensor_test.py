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
output_scope = tf.variable_scope("output")
composor_scope = tf.variable_scope("composor")
leaf_scope = tf.variable_scope("leaf")