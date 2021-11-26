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
				tmp += f"{param}:{test[param]},"
			plt.title(metric)
			rounds = len(tests[str(test)][metric])
			plt.plot([i for i in range(rounds)], tests[str(test)][metric], label=test[variable_metric])
			plt.legend(loc="lower right")
			plt.text(rounds - 1, tests[str(test)][metric][-1], test[variable_metric])
		plt.savefig(f"{output_dest}_{metric}.png", dpi=500)
		plt.show()


if __name__ =='__main__':
	plot_graphs('alpha', 'experiments/oja1.pickle', 'plots/oja1/oja1')