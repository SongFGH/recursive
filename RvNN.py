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
	def __init__(cell, rootNodes, inputs, dtype=None, scope=None):
		if not isinstance(cell, RecursiveCell):
			raise TypeError("cell must be an instance of RecursiveCell")
		if not isinstance(inputs, TreeNode):
			raise TypeError("inputs must be a list")
		if not inputs:
			raise ValueError("inputs must not be empty")

		self.outputs = []
		with variable_scope(scope or "RvNN"):
			batch_size = tf.shape(inputs)
			else
				for root in rootNodes:
					outputs = self.__recursive(cell, root, inputs)

	def __recursive(self, cell, root, inputs):
		if root.isLeaf():
			input_ = inputs[root.frontleaf]
			output, states = cell(input_, None, tf.variable_scope("leaf"));
			self.outputs.append(output)
			return output
		else
			left_output = self.__recursive(cell, root.leftChild, inputs)
			right_output = self.__recursive(cell, root.rightChild, inputs)
			child_states = [left_output, right_output]
			output, states = cell(None, states, tf.variable_scope("composor"));
			self.outputs.append(output)
			return output