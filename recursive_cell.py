import math
import tensorflow as tf
from tensorflow.models.rnn import linear

class RecursiveCell(object):
  """Abstract object representing an recursive cell.

  An recursive cell, in the most abstract setting, is anything that has
  a state -- a vector of floats of size self.state_size -- and performs some
  operation that takes inputs of size self.input_size. This operation
  results in an output of size self.output_size and a new state.

  This module provides a number of basic commonly used recursive cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by a super-class, MultiRNNCell,
  defined later. Every RNNCell must have the properties below and and
  implement __call__ with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this recursive cell on inputs, starting from the given state.

    Args:
      inputs: 2D Tensor with shape [batch_size x self.input_size].
      state: 2D Tensor with shape [batch_size x self.state_size].
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A 2D Tensor with shape [batch_size x self.output_size]
      - New state: A 2D Tensor with shape [batch_size x self.state_size].
    """
    raise NotImplementedError("Abstract method")

  @property
  def input_size(self):
    """Integer: size of inputs accepted by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer: size of state used by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return state tensor (shape [batch_size x state_size]) filled with 0.

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      A 2D Tensor of shape [batch_size x state_size] filled with zeros.
    """
    zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
    # The reshape below is a no-op, but it allows shape inference of shape[1].
    return tf.reshape(zeros, [-1, self.state_size])


class BasicRecursiveCell(RecursiveCell):
  """The most basic Recursive cell."""

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, states, scope=None):
    """Most basic Recursive:
    	leaf:		output = tanh(W * input + Bl).
    	composor:	output = tanh(U * (leftstate, rightstate) + Bc)
    """
    # for a leaf cell
    if inputs not is None and states is None:
    	with tf.variable_scope(scope or type(self).__name__):  # "BasicRecursiveCell"
    		with tf.variable_scope("leaf"):
      			output = tf.tanh(linear.linear(inputs, self._num_units, True))
    			return output, output
    # for a composor cell
    elif inputs is None and states is not None:
    	with tf.variable_scope(scope or type(self).__name__):  # "BasicRecursiveCell"
      		with tf.variable_scope("composor"):
      			output = tf.tanh(linear.linear([states[0], states[1]], self._num_units, True))
    			return output, output
    else
    	raise NotImplementedError("Invalid type of node")