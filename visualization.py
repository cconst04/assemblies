import matplotlib.pyplot as plt
import numpy as np
import random
DIM = 100
class Visualizer:
	def __init__(self):
		self.winners_history = []
		self.index_mappings = {}

	def _map_indices(self, winners):
		free_indices = list(set(range(DIM * DIM)) - set(self.index_mappings.values()))
		np.random.shuffle(free_indices)
		i = 0
		for winner in winners:
			if winner not in self.index_mappings:
				self.index_mappings[winner] = free_indices[i]
				i += 1

	def plot_weights(self, winners, connectome):
		# fill with zeroes to match the lengths of the winners for the next rounds
		# mat = np.pad(mat, (0, MAX_SIZE - mat.shape[0]), mode='constant')
		self._map_indices(winners)
		consecutive_winners = set()
		mapped_winners = [self.index_mappings[winner] for winner in winners]

		if len(self.winners_history) > 0:
			consecutive_winners = set(mapped_winners) & set(self.winners_history[-1])
			overlap_pct = len(consecutive_winners) / float(len(set(mapped_winners).union(set(self.winners_history[-1]))))
		else:
			overlap_pct = 0
		arr_plot = np.zeros((DIM, DIM))
		coords = []
		for winner in mapped_winners:
			coords.append([winner // DIM, winner % DIM, 2 if winner in consecutive_winners else 1])
			# arr_plot[winner // DIM][winner % DIM] = 2 if winner in consecutive_winners else 1
		coords = np.array(coords)
		plt.figure(200)
		plt.clf()
		plt.scatter(coords[:, 0], coords[:, 1], c=coords[:, 2])
		total_subgraph_conn = 0.0
		total_weight = 0.0
		for winner in winners:
			for i in range(0, len(connectome[winner])):
				if connectome[winner][i] > 0.0:
					total_subgraph_conn += 1.0
					total_weight += connectome[winner][i]
					mapped_winner = self.index_mappings[winner]
					mapped_other = self.index_mappings[i]
					plt.plot([mapped_winner//DIM,mapped_other//DIM], [mapped_winner%DIM,mapped_other%DIM], "blue", alpha=connectome[winner][i] / 100.0)
		plt.title(f'{len(consecutive_winners)} overlap: {overlap_pct:.2f} avg conn per neuron: {total_subgraph_conn/len(winners):.2f} avg weight: {total_weight/len(winners):.2f}')
		plt.show(block=False)
		plt.pause(0.1)
		self.winners_history.append(set(mapped_winners))


	def plot_all_potential_winners(self, inputs, smallest_winner_input):
		plt.figure(100)
		plt.clf()
		plt.hist(inputs)
		plt.axvline(x=smallest_winner_input, color='r', linestyle='dashed', linewidth=2)
		plt.show(block=False)
		plt.pause(0.1)