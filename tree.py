import math
from decimal import Decimal, getcontext
import random
from datetime import datetime
import time
import sys

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
		self.depth = None

	def __to_string__(self):
		print "self = " + str(self)
		print "self.l = " + str(self.l)
		print "self.r = " + str(self.r)
		print "self.par = " + str(self.par)
		print "self.value = " + str(self.value)
		print "self.depth = " + str(self.depth)


class Tree():

	def __init__(self, val="", randomness=0, max_depth=7, min_depth=2):
		self.nodes = []
		self.root = Node(val)
		self.root.depth = 0
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
	def gen_rand_tree(randomness=0, max_depth=7, min_depth=1):
		ret = Tree("", randomness, max_depth, min_depth)
		ret.randomize_nodes(ret.root)
		return ret

	def cpy(self):
		ret = Tree()
		ret.node_cpy(self.root, ret.root)
		return ret

	def randomize_nodes(self, node):
		depth = node.depth
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
			self.randomize_nodes(node.l)
			self.randomize_nodes(node.r)
		elif self.operator_par[node.value] == "l":
			self.insert(node, "", "l")
			self.randomize_nodes(node.l)
		elif self.operator_par[node.value] == "r":
			self.insert(node, "", "r")
			self.randomize_nodes(node.r)


	def node_cpy(self, node, target_node):
		target_node.value = node.value

		# Clean up node
		self.delete(target_node.l)
		self.delete(target_node.r)

		# Leaves
		if target_node.value not in self.operator_par.keys():
			return
		# Nonterminal nodes
		elif self.operator_par[target_node.value] == "lr":
			self.insert(target_node, "", "l")
			self.insert(target_node, "", "r")
			self.node_cpy(node.l, target_node.l)
			self.node_cpy(node.r, target_node.r)
		elif self.operator_par[target_node.value] == "l":
			self.insert(target_node, "", "l")
			self.node_cpy(node.l, target_node.l)
		elif self.operator_par[target_node.value] == "r":
			self.insert(target_node, "", "r")
			self.node_cpy(node.r, target_node.r)


	# TODO: implement min depth
	def get_random_node(self, max_depth=0):
		if max_depth <= 0:
			return random.choice(self.nodes)


		node = self.root

		for i in range(max_depth - 1):
			# Leaves
			if node.value not in self.operator_par.keys():
				return node
			# Nonterminal nodes
			elif self.operator_par[node.value] == "lr":
				choice = random.choice([0, 1])
				if choice == 0:
					node = node.l
				else:
					node = node.r
			elif self.operator_par[node.value] == "l":
				node = node.l
			elif self.operator_par[node.value] == "r":
				node = node.r
		return node


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
		self.nodes.append(new)
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

		new.depth = new.par.depth + 1
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
			raise
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

	def print_tree(self):
		str = ""
		thislevel = [self.root]
		depth = 1
		while thislevel:
			space = ""
			for i in range(int(math.ceil((2**self.max_depth)/(2**(depth - 2))))):
				space += "         "
			nextlevel = list()
			for n in thislevel:
				if not n:
					str += space + "None"
				elif n.value != "":
					str += space + n.value
					nextlevel.append(n.l)
					nextlevel.append(n.r)
				else:
					str += space + "Empty",
					nextlevel.append(n.l)
					nextlevel.append(n.r)
			str += "\n"
			depth += 1
			thislevel = nextlevel



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
	tree.randomize_nodes(node)

def crossover(tree, target_tree):
	# Pick random node in tree
	node = random.choice(tree.nodes)

	# Copy node into random node with same depth
	dep = node.depth
	tgt_node = target_tree.get_random_node(dep)
	target_tree.node_cpy(node, tgt_node)

def reproduce(tree):
	return tree.cpy()

def split_tournaments(individual_list, tournament_size):
	result_list = []
	while individual_list:
		result_list.append(individual_list[:tournament_size])
		individual_list = individual_list[tournament_size:]
	return result_list



def tournament(individual_list, number_of_winners, tournament_size):
	# Copy individual list
	copy = list(individual_list)

	# Shuffle copy
	random.shuffle(copy)
	# Split list into tournaments
	tournaments = list(split_tournaments(individual_list, tournament_size))
	winners = []

	if number_of_winners > len(tournaments) or number_of_winners <= 0 or tournament_size <= 0:
		print "Error: there can not be enough winners"
		raise
	# Start tournaments
	for i in range(number_of_winners):
		current_tournament = tournaments[i]
		# Separate invalids from valids
		invalids = [j for j in current_tournament if j[1]]
		valids = [j for j in current_tournament if not j[1]]

		# Sort both
		invalids.sort(key=lambda tup: tup[0])
		valids.sort(key=lambda tup: tup[0])

		# If no valids, chose an invalid
		if not valids:
			winners.append(invalids[0])
		else:
			winners.append(valids[0])

	return winners

def gen_initial_population(size):
	ret = []
	for i in range(size):
		tree = Tree.gen_rand_tree()
		ret.append(tree)

	return ret

def gen_next_population(population, point_set, elitism, tournament_size, mutate_chance, reproduce_chance, crossover_chance):
	# Normalize
	mutate_chance = mutate_chance / (mutate_chance + reproduce_chance + crossover_chance)
	reproduce_chance = reproduce_chance / (mutate_chance + reproduce_chance + crossover_chance)
	crossover_chance = crossover_chance / (mutate_chance + reproduce_chance + crossover_chance)

	num = len(population)

	fitnesses = []

	for i in population:
		fitnesses.append(fitness(i, point_set))

	# Set pos in fitnesses
	for i in range(len(fitnesses)):
		fitnesses[i] = (fitnesses[i][0], fitnesses[i][1], i)

	new_population = []

	best = None

	# Find best
	for i in range(num):
		if best == None or best[1] == True and fitnesses[i][1] == False or best[0] > fitnesses[i][0]:
			best = fitnesses[i]

	if elitism:
		# Keep best
		new_population.append(population[best[2]])
		num -= 1


	# Create new individuals

	for i in range(num):
		roll = random.random()
		if roll >= 0 and roll < mutate_chance:
			new = population[tournament(fitnesses, 1, tournament_size)[0][2]].cpy()
			mutate(new)
			new_population.append(new)
		elif roll >= mutate_chance and roll < mutate_chance + reproduce_chance:
			new = reproduce(population[tournament(fitnesses, 1, tournament_size)[0][2]])
			new_population.append(new)
		elif roll >= mutate_chance + reproduce_chance and roll <= 1:
			winners = tournament(fitnesses, 2, tournament_size)
			new = population[winners[0][2]].cpy()
			crossover(population[winners[1][2]], new)
			new_population.append(new)

	return (best, population[best[2]], new_population)

def gen_points(input_name):
	f = open(input_name, "r")
	points = []
	for i in f:
		strings = [s for s in i.split(" ")]
		point = []
		for j in strings:
			try:
				point.append(float(j))
			except ValueError:
				continue
			if len(point) >= 2:
				break
		points.append(point)
	return points

class Experiment():
	def __init__(self, elitism, tournament_size, 
			mutate_chance, reproduce_chance, crossover_chance,
			pop_size, convergence_factor, max_generations):
		
		self.elitism = elitism
		self.tournament_size = tournament_size
		self.mutate_chance = mutate_chance
		self.reproduce_chance = reproduce_chance
		self.crossover_chance = crossover_chance
		self.pop_size = pop_size
		self.convergence_factor = convergence_factor
		self.max_generations = max_generations
		self.bests = []
		self.pop = []
		self.time = 0

	def run(self, point_set, verbose=False):
		start_time = time.time()

		self.bests = []
		self.pop = gen_initial_population(self.pop_size)
		convergence = 0
		# Last generation is discarded because I am lazy
		for i in range(self.max_generations + 1):
			if verbose: print "Generating population ", i
			self.pop = gen_next_population(self.pop, point_set, self.elitism, self.tournament_size,
				self.mutate_chance, self.reproduce_chance, self.crossover_chance)
			last_best = (self.pop[0], self.pop[1])
			self.pop = self.pop[2]
			self.bests.append(last_best)
			if i > 0 and self.bests[i][0][0] == self.bests[i - 1][0][0]:
				convergence += 1
				# Converged
				if convergence >= self.convergence_factor:
					break
			else:
				convergence = 0


		self.time = time.time() - start_time

	def output(self, data_filename, stats_filename):
		data = open(data_filename, "w")
		stats = open(stats_filename, "w")

		for i in self.bests:
			data.write(i[1].to_string() + "\n")
			data.write(i[1].print_tree() + "\n")
			data.write(str(i[0]) + "\n")

		stats.write("elitism: " + str(self.elitism) + "\n")
		stats.write("tournament_size: " + str(self.tournament_size) + "\n")
		stats.write("mutate_chance: " + str(self.mutate_chance) + "\n")
		stats.write("reproduce_chance: " + str(self.reproduce_chance) + "\n")
		stats.write("crossover_chance: " + str(self.crossover_chance) + "\n")
		stats.write("pop_size: " + str(self.pop_size) + "\n")
		stats.write("convergence_factor: " + str(self.convergence_factor) + "\n")
		stats.write("max_generations: " + str(self.max_generations) + "\n")
		stats.write("time: " + str(self.time) + "\n")









for i in range(5):
	for j in range(10):

		print "---------------------------------------------------------------"
		pars = []

		# elitism, tournament_size, mutate_chance, reproduce_chance, crossover_chance,
		# pop_size, convergence_factor, max_generations

		# Base test
		if i == 0:
			pars = [True, 5, 5, 5, 90, 10, 25, 100]
		# Remove elitism
		elif i == 1:
			pars = [False, 5, 5, 5, 90, 10, 25, 100]
		# Increase tournament 1
		elif i == 2:
			pars = [True, 5, 5, 5, 90, 10, 25, 100]
		# Increase tournament 2
		elif i == 3:
			pars = [True, 10, 5, 5, 90, 10, 25, 100]
		# Increase tournament 3
		elif i == 4:
			pars = [True, 15, 5, 5, 90, 10, 25, 100]
		# Change probability 1
		elif i == 5:
			pars = [True, 5, 20, 5, 75, 10, 25, 100]
		# Change probability 2
		elif i == 6:
			pars = [True, 5, 45, 5, 50, 10, 25, 100]
		# Increase pop 1
		elif i == 6:
			pars = [True, 5, 5, 5, 90, 10, 50, 100]
		# Increase pop 2
		elif i == 6:
			pars = [True, 5, 5, 5, 90, 10, 100, 100]
		# Increase convergence
		elif i == 6:
			pars = [True, 5, 5, 5, 90, 20, 25, 100]
		# Increase generations
		elif i == 6:
			pars = [True, 5, 5, 5, 90, 20, 25, 200] 






		exp = Experiment(pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7])
		exp.run(gen_points("datasets-TP1/SR_circle.txt"), True)
		exp.output("outputs/data" + str(i) + "." + str(j) + "_circle.txt", "outputs/stats" + str(i) + "." + str(j) + "_circle.txt")

		exp = Experiment(pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7])
		exp.run(gen_points("datasets-TP1/SR_div.txt"), True)
		exp.output("outputs/data" + str(i) + "." + str(j) + "_div.txt", "outputs/stats" + str(i) + "." + str(j) + "_div.txt")

		exp = Experiment(pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7])
		exp.run(gen_points("datasets-TP1/SR_elipse_noise.txt"), True)
		exp.output("outputs/data" + str(i) + "." + str(j) + "_elipse_noise.txt", "outputs/stats" + str(i) + "." + str(j) + "_elipse_noise.txt")

		exp = Experiment(pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7])
		exp.run(gen_points("datasets-TP1/SR_div_noise.txt"), True)
		exp.output("outputs/data" + str(i) + "." + str(j) + "_div_noise.txt", "outputs/stats" + str(i) + "." + str(j) + "_div_noise.txt")
