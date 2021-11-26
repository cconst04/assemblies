import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt

from collections import OrderedDict
from visualization import Visualizer

import pickle
import itertools

def project_test(n=100000,k=317,p=0.01,beta=0.05, alpha=None, punish_beta=None, reward_ratio=1/2, rounds=100, learning_rule='hebb'):
	viz = Visualizer()
	b = brain.Brain(p)
	b.add_stimulus("stim",k)
	b.add_area("A", n, k, beta, learning_rule=learning_rule, time_function='step', punish_beta=punish_beta, reward_ratio=reward_ratio, alpha=alpha)
	# b.add_area("A", n, k, beta, learning_rule='stdp', time_function='step', punish_beta=beta/2)
	# b.add_area("A", n, k, beta, learning_rule='oja')
	# b.add_area("A", n, k, beta, learning_rule='hebb')
	b.project({"stim":["A"]},{})
	viz.plot_weights(b.areas['A'].winners)
	print(f'n:{n}, k:{k}, p:{p}, beta:{beta}, punish_beta:{punish_beta}, reward_ratio:{reward_ratio}')
	result_stats = {'support':[], 'overlap_pct':[]}
	for i in range(rounds):
		print(f'round:{i}')
		b.project({"stim":["A"]},{"A":["A"]})
		# print(b.get_winner_weights_stats())
		viz.plot_weights(b.areas['A'].winners)
		stats = b.get_winner_weights_stats()
		for stat in stats:
			if not stat in result_stats:
				result_stats[stat] = []
			result_stats[stat].append(stats[stat])
		result_stats['support'].append(b.areas['A'].w)
		result_stats['overlap_pct'].append(viz.overlap_pct)
		print(f"support:{b.areas['A'].w}, pct:{viz.overlap_pct}")
	return result_stats




def fixed_assembly_test(n=100000,k=317,p=0.01,beta=0.05):
	viz = Visualizer()
	b = brain.Brain(p)
	b.add_stimulus("stim",k)
	b.add_area("A", n, k, beta, learning_rule='hebb')
	# b.add_area("A", n, k, beta, learning_rule='stdpv2', time_function='step')
	b.project({"stim":["A"]},{})
	viz.plot_weights(b.areas['A'].winners)
	for i in range(100):
		b.project({"stim":["A"]},{"A":["A"]})
		viz.plot_weights(b.areas['A'].winners)
		print((b.areas["A"].w))
	b.areas["A"].fix_assembly()
	for i in range(5):
		b.project({"stim":["A"]},{"A":["A"]})
		print((b.areas["A"].w))
	b.areas["A"].unfix_assembly()
	for i in range(5):
		b.project({"stim":["A"]},{"A":["A"]})
		print((b.areas["A"].w))

def explicit_assembly_test():
	b = brain.Brain(0.5)
	b.add_stimulus("stim",3)
	b.add_explicit_area("A",10,3,beta=0.5, learning_rule='hebb')
	b.add_area("B",10,3,beta=0.5)
	print((b.stimuli_connectomes["stim"]["A"]))
	print((b.connectomes["A"]["A"]))
	print((b.connectomes["A"]["B"].shape))
	print((b.connectomes["B"]["A"].shape))

	# Now test projection stimulus -> explicit area
	print("Project stim->A")
	# b.project({"stim":["A"]},{})
	print((b.areas["A"].winners))
	print((b.stimuli_connectomes["stim"]["A"]))
	# Now test projection stimulus, area -> area
	for _ in range(20):
		b.project({"stim":["A"]},{"A":["A"]})
	print((b.areas["A"].winners))
	print((b.stimuli_connectomes["stim"]["A"]))
	print((b.connectomes["A"]["A"]))

	# project explicit A -> B
	print("Project explicit A -> normal B")
	b.project({},{"A":["B"]})
	print((b.areas["B"].winners))
	print((b.connectomes["A"]["B"]))
	print((b.connectomes["B"]["A"]))
	print((b.stimuli_connectomes["stim"]["B"]))

def explicit_assembly_test2(rounds=20):
	b = brain.Brain(0.1)
	b.add_explicit_area("A",100,10,beta=0.5)
	b.add_area("B",10000,100,beta=0.5)

	b.areas["A"].winners = list(range(10,20))
	b.areas["A"].fix_assembly()
	b.project({}, {"A": ["B"]})

	# Test that if we fire back from B->A now, we don't recover the fixed assembly
	b.areas["A"].unfix_assembly()
	b.project({}, {"B": ["A"]})
	print((b.areas["A"].winners))

	b.areas["A"].winners = list(range(10,20))
	b.areas["A"].fix_assembly()
	b.project({}, {"A": ["B"]})
	b.project({}, {"A": ["B"], "B": ["A", "B"]})
	print((b.areas["B"].w))

	b.areas["A"].unfix_assembly()
	b.project({}, {"B": ["A"]})
	print("After 1 B->A, got A winners:")
	print((b.areas["A"].winners))

	for _ in range(4):
		b.project({}, {"B": ["A"], "A": ["A"]})
	print("After 5 B->A, got A winners:")
	print((b.areas["A"].winners))

def explicit_assembly_recurrent():
	b = brain.Brain(0.1)
	b.add_explicit_area("A",100,10,beta=0.5)

	b.areas["A"].winners = list(range(60,70))


def run_tests(params, fname):
	"""
	Varying alpha and keeping all the other params constant
	"""
	total_combinations = 1
	for param in params:
		total_combinations *= len(params[param])
	print(f'total_combinations:{total_combinations}')
	res = {}
	# itertools.product(params.values())
	completed_so_far = 0
	for comb in itertools.product(*params.values()):
		print(comb)
		kwargs = {}
		for i in range(len(comb)):
			kwargs[list(params.keys())[i]] = comb[i]
		stats = project_test(**kwargs)
		res[str(kwargs)] = stats
	# Store data (serialize)
		with open(fname, 'wb') as handle:
		    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
		# Load data (deserialize)
		print(f'{completed_so_far}/{total_combinations}')
		completed_so_far += 1




if __name__ == '__main__':
	# params = {
	# 	'p': [0.001, 0.003, 0.007, 0.015, 0.031, 0.063, 0.127, 0.255, 0.511],
	# 	'k': [i for i in range(100, 1000, 100)],
	# 	'alpha': [0.001, 0.003, 0.007, 0.015, 0.031, 0.063, 0.127, 0.255, 0.511],
	# 	'beta': [0.001, 0.003, 0.007, 0.015, 0.031, 0.063, 0.127, 0.255, 0.511],
	# 	'punish_beta': [0, 0.001, 0.003, 0.007, 0.015, 0.031, 0.063, 0.127, 0.255, 0.511],
	# 	'reward_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],
	# 	'learning_rule':['hebb', 'stdp', 'stdpv2', 'oja']

	# }
	#oja test 1
	# params = {
	# 	'p': [0.01],
	# 	'k': [300],
	# 	'alpha': [0.001, 0.002, 0.003, 0.005, 0.008, 0.01],
	# 	'beta': [0.05],
	# 	'punish_beta': [0.01],
	# 	'reward_ratio': [0.1],
	# 	'learning_rule':['oja']

	# }
	# run_tests(params, 'experiments/oja1.pickle')
	# params = {
	# 	'p': [0.01],
	# 	'k': [300],
	# 	'alpha': [0.01],
	# 	'beta': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
	# 	'punish_beta': [None],
	# 	'reward_ratio': [0.5],
	# 	'learning_rule':['stdp']

	# }
	# run_tests(params, 'experiments/stdp_1.pickle')
	params = {
		'p': [0.01],
		'k': [300],
		'alpha': [0.01],
		'beta': [0.05],
		'punish_beta': [0.025],
		'reward_ratio': [0.9, 0.8, 0.7, 0.6, 0.5],
		'learning_rule':['oja']

	}
	run_tests(params, 'experiments/stdpv2_1.pickle')


