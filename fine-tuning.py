from gridworld import GridWorldMDP
from qlearn import QLearner

import numpy as np
import matplotlib.pyplot as plt

import random

def plot_convergence(utility_grids, policy_grids):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
	ax1.plot(utility_ssd, 'b.-')
	ax1.set_ylabel('Change in Utility', color='b')

	policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
	ax2.plot(policy_changes, 'r.-')
	ax2.set_ylabel('Change in Best Policy', color='r')


if __name__ == '__main__':
	shape = (6, 8)
	goal = (5, -1)
	trap1 = (1, -1)
	trap2 = (4, 1)
	trap3 = (4, 2)
	trap4 = (4, 3)
	trap5 = (4, 4)
	trap6 = (4, 5)
	trap7 = (4, 6)
	obstacle1 = (1, 1)
	obstacle2 = (0, 5)
	obstacle3 = (2, 3)
	start = (2, 0)
	obstacle4 = (3, 5)
	default_reward = -0.1
	goal_reward = 1
	trap_reward = -1

	reward_grid = np.zeros(shape) + default_reward
	reward_grid[goal] = goal_reward
	reward_grid[trap1] = trap_reward
	reward_grid[trap2] = trap_reward
	reward_grid[trap3] = trap_reward
	reward_grid[trap4] = trap_reward
	reward_grid[trap5] = trap_reward
	reward_grid[trap6] = trap_reward
	reward_grid[trap7] = trap_reward
	reward_grid[obstacle1] = 0
	reward_grid[obstacle2] = 0
	reward_grid[obstacle3] = 0
	reward_grid[obstacle4] = 0

	terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
	terminal_mask[goal] = True
	terminal_mask[trap1] = True
	terminal_mask[trap2] = True
	terminal_mask[trap3] = True
	terminal_mask[trap4] = True
	terminal_mask[trap5] = True
	terminal_mask[trap6] = True
	terminal_mask[trap7] = True

	obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
	obstacle_mask[1, 1] = True
	obstacle_mask[0, 5] = True
	obstacle_mask[2, 3] = True
	obstacle_mask[3, 5] = True
	
	bestValue = - 10
	best_learning_rate = None
	best_discount_rate = None
	best_random_action_prob = None
	best_random_action_decay_rate = None
	i = 0
	while True:
		learning_rate = random.random()
		discount_rate = random.random()
		random_action_prob = random.random()
		random_action_decay_rate = 0.99 + random.random()/100
		
		gw = GridWorldMDP(reward_grid=reward_grid,
						  obstacle_mask=obstacle_mask,
						  terminal_mask=terminal_mask,
						  action_probabilities=[
							  (-1, 0.1),
							  (0, 0.8),
							  (1, 0.1),
						  ],
						  no_action_probability=0.0)

		ql = QLearner(num_states=(shape[0] * shape[1]),
					  num_actions=4,
					  learning_rate=learning_rate,
					  discount_rate=discount_rate,
					  random_action_prob=random_action_prob,
					  random_action_decay_rate=random_action_decay_rate,
					  dyna_iterations=0)

		start_state = gw.grid_coordinates_to_indices(start)	
		iterations = 1000
		flat_policies, flat_utilities = ql.learn(start_state,
												 gw.generate_experience,
												 iterations=iterations)


		test_iterations = 1000
		value = ql.test(start_state,gw.generate_experience,iterations=test_iterations)
		
		if value>=bestValue:
			if value == 1000:
				print("BEST FOUND")
				# f = open("best_parameters.txt",'a')
				# f.write("expriment: " +str(i) +'\n')
				# f.write("learning_rate: " +str(learning_rate) +'\n')
				# f.write("discount_rate: " +str(discount_rate) +'\n')
				# f.write("random_action_prob: " +str(random_action_prob) +'\n')
				# f.write("random_action_decay_rate: " +str(random_action_decay_rate) +'\n')
				# f.close()
				new_shape = (gw.shape[0], gw.shape[1], iterations)
				ql_utility_grids = flat_utilities.reshape(new_shape)
				ql_policy_grids = flat_policies.reshape(new_shape)
				plt.figure()
				gw.plot_policy(ql_utility_grids[:, :, -1], ql_policy_grids[:, :, -1])
				plot_convergence(ql_utility_grids, ql_policy_grids)
				plt.show()
				input()
				
			bestValue = value
			best_learning_rate = learning_rate
			best_discount_rate = discount_rate
			best_random_action_prob = random_action_prob
			best_random_action_decay_rate = random_action_decay_rate
		
		i+=1
		print()
		print("expriment:",i)
		
		print("current value:",value)
		print(learning_rate)
		print(discount_rate)
		print(random_action_prob)
		print(random_action_decay_rate)
		
		print("best value:",bestValue)
		print(best_learning_rate)
		print(best_discount_rate)
		print(best_random_action_prob)
		print(best_random_action_decay_rate)