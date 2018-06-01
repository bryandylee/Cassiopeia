import networkx as nx
import numpy as np

def node_parent(x, y):
	"""
	Given two nodes, finds the latest common ancestor

	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:return:
		Returns latest common ancestor of x and y
	"""

	parr = []
	x_list = x.split('|')
	y_list = y.split('|')
	for i in range(0,len(x_list)):
		if x_list[i] == y_list[i]:
			parr.append(x_list[i])
		elif x_list[i] == '-':
			parr.append(y_list[i])
		elif y_list[i] == '-':
			parr.append(x_list[i])
		else:
			parr.append('0')

	return '|'.join(parr)

def get_edge_length(x,y,priors=None):
	"""
	Given two nodes, if x is a parent of y, returns the edge length between x and y, else -1

	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:param priors:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		Length of edge if valid transition, else -1
	"""
	count = 0
	x_list = x.split('|')
	y_list = y.split('|')

	for i in range(0, len(x_list)):
			if x_list[i] == y_list[i]:
					pass
			elif y_list[i] == "-":
					count += 0

			elif x_list[i] == '0':
				if not priors:
					count += 1
				else:
					count += - np.log(priors[i][str(y_list[i])])
			else:
				return -1
	return count

def root_finder(target_nodes):
	"""
	Given a list of targets_nodes, return the least common ancestor of all nodes

	:param target_nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		The least common ancestor of all target nodes, in the form 'Ch1|Ch2|....|Chn'
	"""
	np = target_nodes[0]
	for sample in target_nodes:
		np = node_parent(sample, np)

	return np

def build_potential_graph_from_base_graph(samples, priors=None):
	"""
	Given a series of samples, or target nodes, creates a tree which contains potential
	ancestors for the given samples.

	First, a directed graph is constructed, by considering all pairs of samples, and checking
	if a sample can be a possible parent of another sample
	Then we all pairs of nodes with in-degree 0 and < a certain edit distance away
	from one another, and add their least common ancestor as a parent to these two nodes. This is done
	until only one possible ancestor remains

	:param samples:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param priors
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		A graph, which contains a tree which explains the data with minimal parsimony
	"""
	#print "Initial Sample Size:", len(set(samples))
	initial_network = nx.DiGraph()
	samples = set(samples)
	for sample in samples:
		initial_network.add_node(sample)

	samples = list(samples)
	for i in range(0, len(samples)):
		sample = samples[i]
		for j in range(i+1, len(samples)):
			sample_2 = samples[j]
			edge_length = get_edge_length(sample, sample_2)
			if edge_length != -1:
				initial_network.add_edge(sample, sample_2, weight=edge_length)

	source_nodes = get_sources_of_graph(initial_network)

	print "Number of initial extrapolated pairs:", len(source_nodes)
	while len(source_nodes) != 1:
		temp_source_nodes = set()
		for i in range(0, len(source_nodes)-1):
			sample = source_nodes[i]
			top_parents = []
			for j in range(i + 1, len(source_nodes)):
				sample_2 = source_nodes[j]
				if sample != sample_2:
					parent = node_parent(sample, sample_2)
					top_parents.append((get_edge_length(parent, sample) + get_edge_length(parent, sample_2), parent, sample_2))

					# Check this cutoff
					if get_edge_length(parent, sample) + get_edge_length(parent, sample_2) < 4:
						initial_network.add_edge(parent, sample_2, weight=get_edge_length(parent, sample_2, priors))
						initial_network.add_edge(parent, sample, weight=get_edge_length(parent, sample, priors))
						temp_source_nodes.add(parent)

			min_distance = min(top_parents, key = lambda k: k[0])[0]
			lst = [(s[1], s[2]) for s in top_parents if s[0] <= min_distance]

			for parent, sample_2 in lst:
				initial_network.add_edge(parent, sample_2, weight=get_edge_length(parent, sample_2, priors))
				initial_network.add_edge(parent, sample, weight=get_edge_length(parent, sample, priors))
				temp_source_nodes.add(parent)

		source_nodes = list(temp_source_nodes)

		print "Next layer number of nodes:", len(source_nodes)

	return initial_network


def get_sources_of_graph(tree):
	"""
	Returns all nodes with in-degree zero

	:param tree:
		networkx tree
	:return:
		Leaves of the corresponding Tree
	"""
	return [x for x in tree.nodes() if tree.in_degree(x)==0 ]
