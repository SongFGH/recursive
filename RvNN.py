import TreeNode
from recursive_cell import RecursiveCell, BasicRecursiveCell

import tensorflow as tf
from tensorflow.models.rnn import linear

class RvNN:
	'''
  	Args:
    	cell: An instance of RNNCell.
    	rootNodes: A length T list of rootNodes , each a type of TreeNode.
    	inputs: the word vectors of words batch_size sentences, with a
    		list in length batch_size, in which each element is a array with
    		shape [word_number, wordvec_dim], where word_number differs in
    		every sentence.
    	initial_state: (optional) An initial state for the RNN.  This must be
      		a tensor of appropriate type and shape [batch_size x cell.state_size].
    	dtype: (optional) The data type for the initial state.  Required if
      		initial_state is not provided.
    	scope: VariableScope for the created subgraph; defaults to "RNN".

  	Returns:
    	A pair (outputs, states) where:
      		outputs is a length T list of outputs (one for each input)
      		states is a length T list of states (one state following each input)

  	Raises:
    	TypeError: If "cell" is not an instance of RNNCell.
    	ValueError: If inputs is None or an empty list.
    '''
	def __init__(cell, rootNodes, inputs, initial_state=None, dtype=None, scope=None):
		if not isinstance(cell, RecursiveCell):
			raise TypeError("cell must be an instance of RecursiveCell")
		if not isinstance(inputs, TreeNode):
			raise TypeError("inputs must be a list")
		if not inputs:
			raise ValueError("inputs must not be empty")

		self.output = []
		self.states = []
		with variable_scope(scope or "RvNN"):
			batch_size = tf.shape(inputs[0])[0]
			if initial_state is not None:
				state = initial_state
			else
				if not dtype:
					raise ValueError("If no initial_state is provided, dtype must be.")
				state = cell.zero_state(batch_size, dtype)

			if sequence_length:
				zero_output_state = (
					tf.zeros(tf.pack([batch_size, cell.output_size]), inputs[0].dtype),
					tf.zeros(tf.pack([batch_size, cell.state_size])))