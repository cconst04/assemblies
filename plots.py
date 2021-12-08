import pickle
import matplotlib.pyplot as plt
def plot_graphs(variable_metric, fname, output_dest):
	"""
	variable metric: the metric that varies throughout the experiment
	fname: the pickle file source path
	output_dest: the output plots destination
	"""
	with open(fname, 'rb') as handle:
		tests = pickle.load(handle)
	all_metrics = tests[list(tests.keys())[0]].keys()
	for metric in all_metrics:
		for test in tests:
			tmp = ''
			test = eval(test)
			for param in test:
				if param in ['alpha', 'k']:
					tmp += f"{param}:{test[param]},"
				# tmp += f"{param}:{test[param]},"
			plt.title(metric)
			rounds = len(tests[str(test)][metric])
			plt.plot([i for i in range(rounds)], tests[str(test)][metric], label=tmp)
			plt.legend(loc="lower right")
			plt.text(rounds - 1, tests[str(test)][metric][-1], test[variable_metric])
		plt.savefig(f"{output_dest}_{metric}.png", dpi=500)
		plt.show()


def merge_pickles(path1, path2):
	with open(path1, 'rb') as handle:
		tests1 = pickle.load(handle)
	with open(path2, 'rb') as handle:
		tests2 = pickle.load(handle)
	for params in tests2:
		tests1[params] = tests2[params]
	with open(path1, 'wb') as handle:
		pickle.dump(tests1, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ =='__main__':
	plot_graphs('punish_beta', 'experiments/stdp_1.pickle', 'plots/stdp_1/stdp_1')
	# merge_pickles('experiments/oja1.pickle', 'experiments/hebb.pickle')

