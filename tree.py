import math
from decimal import Decimal, getcontext
import random
from datetime import datetime

# Set seed as time
random.seed(datetime.now())

def uniqueify(my_list):
	return [e for i, e in enumerate(my_list) if my_list.index(e) == i]

class Ops():

	# Basic operations

	@staticmethod
	def add(x, y):
		return [x+y]
	@staticmethod
	def sub(x, y):
		return [x-y]
	@staticmethod
	def div(x, y):
		return [x/y]
	@staticmethod
	def mul(x, y):
		return [x*y]
	@staticmethod
	def addsub(x, y):
		return [(x+y), (x-y)]

	# Power methods

	@staticmethod
	def power(x, y):
		return [x**y]
	@staticmethod
	def root(x, y):
		[x**(1/y)]
	@staticmethod
	def log(x, y):
		return [math.log(x, base=y)]



	# Trigonometric operations

	@staticmethod
	def sin(nothing, x):
		return [math.sin(x)]
	@staticmethod
	def cos(nothing, x):
		return [math.cos(x)]
	@staticmethod
	def tan(nothing, x):
		return [math.tan(x)]
	@staticmethod
	def asin(nothing, x):
		return [math.asin(x)]
	@staticmethod
	def acos(nothing, x):
		return [math.acos(x)]
	@staticmethod
	def atan(nothing, x):
		return [math.atan(x)]



class Node():
	def __init__(self, val):
		self.l = None
		self.r = None
		self.par = None
		self.value = val

class Tree():

	def __init__(self, val, randomness=0, max_depth=7, min_depth=2):
		self.nodes = []
		self.root = Node(val)
		self.nodes.append(self.root)
		self.value_randomness = randomness
		self.max_depth = max_depth
		self.min_depth = min_depth
		self.operators = { "+": Ops.add,
		                   "-": Ops.sub,
		                   "/": Ops.div,
		                   "*": Ops.mul,
		                   "+-": Ops.addsub,
		                   "^": Ops.power,
		                   "sin": Ops.sin,
		                   "cos": Ops.cos,
		                   "tan": Ops.tan }
		self.operator_par = { "+": "lr",
		                      "-": "lr",
		                      "/": "lr",
		                      "*": "lr",
		                      "+-": "lr",
		                      "^": "lr",
		                      "sin": "r",
		                      "cos": "r",
		                      "tan": "r" }



	@staticmethod
	def gen_rand_tree():
		ret = Tree("")
		ret.randomize_nodes(ret.root)
		return ret

	def randomize_nodes(self, node, depth=0):
		rand_ops = self.operators.keys()
		rand_ops.extend(["x", "val"])
		new_node_op = random.choice(rand_ops)
		if depth == self.max_depth:
			new_node_op = random.choice(["x", "val"])
		if depth < self.min_depth:
			new_node_op = random.choice(self.operators.keys())


		node.value = new_node_op

		# Clean up node
		self.delete(node.l)
		self.delete(node.r)

		# Leaves
		if new_node_op not in self.operator_par.keys() and new_node_op != "val":
			return
		elif new_node_op == "val":
			value = random.random()
			# Each time you divide, you get a more random distribution
			# (the distribution tends to generate ever higher values)
			for i in range(self.value_randomness):
				value /= random.random() + 0.1
			node.value = str(value)

		# Nonterminal nodes
		elif self.operator_par[node.value] == "lr":
			self.insert(node, "", "l")
			self.insert(node, "", "r")
			self.randomize_nodes(node.l, depth + 1)
			self.randomize_nodes(node.r, depth + 1)
		elif self.operator_par[node.value] == "l":
			self.insert(node, "", "l")
			self.randomize_nodes(node.l, depth + 1)
		elif self.operator_par[node.value] == "r":
			self.insert(node, "", "r")
			self.randomize_nodes(node.r, depth + 1)

	def get_depth(self, node):
		depth = 0
		while (node.par != None):
			node = node.par
			depth += 1

		return depth

	def insert(self, node, val, pos):
		if node not in self.nodes:
			print "Error: parent node does not exist"
			raise
		new = Node(val)
		self.nodes.append(Node(val))
		new.par = node
		if pos == "l":
			if node.l != None:
				print "Error: node left not empty"
				raise
			node.l = new

		elif pos == "r":
			if node.r != None:
				print "Error: node right not empty"
				raise
			node.r = new

		self.nodes.append(new)
		return new

	# Delete node and children
	def delete(self, node):
		if node == None:
			return
		self.nodes.remove(node)
		if node.par.l == node:
			node.par.l = None
		elif node.par.r == node:
			node.par.r = None
		else:
			print "Error: node structure is corrupted"
			exit()
		self.delete(node.l)
		self.delete(node.r)

		if node == self.root:
			self.root = None

	def __to_string__(self, node):

		if (node == None):
			return ""

		string_l = self.__to_string__(node.l)
		string_r = self.__to_string__(node.r)

		if node.value not in self.operator_par.keys():
			string = node.value

		elif self.operator_par[node.value] == "lr":
			string = " (" + string_l + ") " + node.value + " (" + string_r + ") "

		elif self.operator_par[node.value] == "l":
			string = " (" + string_l + ") " + node.value

		elif self.operator_par[node.value] == "r":
			string = node.value + " (" + string_r + ") "


		return string

	def to_string(self):
		return self.__to_string__(self.root)

	#def from_string(string):

	# Can evaluate nonfunctions
	def __eval__(self, x, node):
		if node == None:
			return [None]

		if node.value == "x":
			return [x]

		try:
			return [float(node.value)]
		except ValueError:
			l = self.__eval__(x, node.l)
			r = self.__eval__(x, node.r)
			ret = []
			for i in l:
				for j in r:			
					try:
						ret.extend(self.operators[node.value](i, j))
					except ValueError:
						ret.append(None)
					except TypeError:
						ret.append(None)
					except ZeroDivisionError:
						ret.append(None)
					except OverflowError:
						ret.append(None)
				# End for
			# End for
			return uniqueify(ret)


	def eval(self, x):
		return self.__eval__(x, self.root)



# Genetic algorithm helper functions start here #

def get_error(tree, x, y):
	best = None
	for i in tree.eval(x):
		if best != None:
			if i != None:
				if best > abs(y - i):
					best = abs(y - i)
		elif i != None:
			best = abs(y - i)
	return best

def fitness(tree, point_list):
	none = False
	n = len(point_list)
	p = 0
	for i in point_list:
		j = get_error(tree, i[0], i[1])
		if j != None:
			p += j**2
		else:
			none = True
			n -= 1

	if n == 0:
		return (None, none)
	return (p/n, none)

def mutate(tree):
	node = random.choice(tree.nodes)
	tree.randomize_nodes(node, tree.get_depth(node))

tree = Tree.gen_rand_tree()
print tree.to_string()
print tree.eval(1)
print fitness(tree, [[1, 2], [2, 3], [3, 4]])
mutate(tree)
print tree.to_string()
print tree.eval(1)
print fitness(tree, [[1, 2], [2, 3], [3, 4]])
