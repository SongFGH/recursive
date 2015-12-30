import TreeNode
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class RvNN:
	def __init__(self, input_dim, hidden_dim, output_dim, epoch=5, learning_rate=1e-4, batch_size=20, num_layers=1, dropout=False):
		self.batch_size = batch_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.dropout = dropout

	def model(self, rootNode):
		self.num_layers = 2


if __name__ == '__main__':
	rvnn = RvNN(300, 150, 5, batch_size=10)
	rvnn.model(None)
	print(rvnn.num_layers, rvnn.input_dim, rvnn.batch_size, rvnn.learning_rate)