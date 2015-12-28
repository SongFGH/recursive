class TreeNode(object):
	def __init__(self, nid, frontLeaf):
		self.nid = nid
		self.label = 0
		self.reln = 0
		self.frontLeaf = frontLeaf
		self.leftChild = None
		self.rightChild = None
		self.parent = None

	def setParent(self, parentId, nodes):
		if nodes[parentId] is None:
			nodes[parentId] = TreeNode(parentId, self.frontLeaf)
		nodes[parentId].addChild(self)
		self.parent = nodes[parentId]

	def addChild(self, node):
		if self.leftChild is None:
			self.leftChild = node
			self.frontLeaf = node.frontLeaf
		elif self.frontLeaf > node.frontLeaf:
			self.rightChild = self.leftChild
			self.leftChild = node
			self.frontLeaf = node.frontLeaf
		else:
			self.rightChild = node

	def isLeaf(self):
		return self.leftChild is None

	def printNode(self):
		if self.parent is not None:
			print('node-', self.nid, '(Leaf-', self.isLeaf(), '): ', self.parent.nid)
		else:
			print('node-', self.nid, '(ROOT)')