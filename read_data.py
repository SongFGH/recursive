from TreeNode import *

def loadTree(parent_str, label_str, relns_str):
	parents = parent_str.split(' ')
	labels = label_str.split(' ')
	relns = relns_str.split(' ')
	nodes = [None] * len(parents)

	i = 0
	for parent in parents:
		parent = int(parent)
		label = int(labels[i])
		reln = int(relns[i])
		if nodes[i] is None:
			nodes[i] = TreeNode(i, i)
		nodes[i].label = label
		nodes[i].reln = reln
		if parent != 0:
			nodes[i].setParent(parent-1, nodes)
		i = i + 1

	for node in nodes:
		node.printNode()

parent_str = '70 70 68 67 63 62 61 60 58 58 57 56 56 64 65 55 54 53 52 51 49 47 47 46 46 45 40 40 41 39 38 38 43 37 37 69 44 39 42 41 42 43 44 45 50 48 48 49 50 51 52 53 54 55 66 57 59 59 60 61 62 63 64 65 66 67 68 69 71 71 0'
label_str = '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 1 1 1 2 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 2 1 2 0 1'
relns_str = '1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3'
loadTree(parent_str, label_str, relns_str)
