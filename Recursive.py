import TreeNode
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class Recursive(object):
	def __init__(self, batch_size=20, input_dim=300, hidden_dim=150, output_dim=5, learning_rate=1e-4, num_layers=1):
		self.batch_size = batch_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.learning_rate = learning_rate

	def model(self, rootNode):

	def train(self, data_batch):

	def predict(self, data_batch):