import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


class GridWorldMDP:

	# up, right, down, left
	_direction_deltas = [
		(-1, 0),
		(0, 1),
		(1, 0),
		(0, -1),
	]
	# the number of possible actions
	_num_actions = len(_direction_deltas)

	def __init__(self,
				reward_grid,
				terminal_mask,
				obstacle_mask,
				action_probabilities,
				no_action_probability):

		# 6 * 8 array, reward
		self._reward_grid = reward_grid
		# 6 * 8 array, terminal: goal/trap
		self._terminal_mask = terminal_mask
		# 6 * 8 array, obstacle
		self._obstacle_mask = obstacle_mask
		
		# Matrix _T is a transition matrix when you provide row, col, num_action
		# it will return back a probability matrix which encodes the probability 
		# Ex.
		# _T[0,0,1,:,:]	is
		#
		# [[0.1 0.8 0.	0.	0.	0.	0.	0. ]
		#	[0.1 0.	0.	0.	0.	0.	0.	0. ]
		#	[0.	0.	0.	0.	0.	0.	0.	0. ]
		#	[0.	0.	0.	0.	0.	0.	0.	0. ]
		#	[0.	0.	0.	0.	0.	0.	0.	0. ]
		#	[0.	0.	0.	0.	0.	0.	0.	0. ]]
		#
		# It means in position (0,0), choose action 1 (right)
		# With possibility 0.1 stays in place, possibility 0.8 moves to the right and possibility 0.1 moves down
		self._T = self._create_transition_matrix(
			action_probabilities,
			no_action_probability,
			obstacle_mask
		)

	@property
	def shape(self):
		return self._reward_grid.shape

	@property
	def size(self):
		return self._reward_grid.size

	@property
	def reward_grid(self):
		return self._reward_grid

	def run_value_iterations(self, discount=1.0,
						iterations=10):
		# utility_grids and policy_grids is 6 * 8 * 10, all value is zero
		utility_grids, policy_grids = self._init_utility_policy_storage(iterations)

		# utility_grid is 6 * 8, all value is zero
		utility_grid = np.zeros_like(self._reward_grid)
		for i in range(iterations):
			utility_grid = self._value_iteration(utility_grid=utility_grid,discount = discount)
			policy_grids[:, :, i] = self.best_policy(utility_grid)
			utility_grids[:, :, i] = utility_grid
		return policy_grids, utility_grids

	def run_policy_iterations(self, discount=1.0, iterations=10):
		utility_grids, policy_grids = self._init_utility_policy_storage(iterations)
		
		# initialization
		M, N = self.shape 
		policy_grid = np.random.randint(0, self._num_actions,self.shape)
		utility_grid = np.random.rand(M,N)/100
		
		for i in range(iterations):
			# policy evaluation
			utility_grid = self._reward_grid + discount * (
				# probability * v(s)
				(utility_grid.reshape((1,1,1,M,N)) * self._T
				).sum(axis=-1).sum(axis=-1)
				* 
				# one-hot of policy_grid
				(np.arange(self._num_actions) == policy_grid.reshape((M,N,1))).astype(np.int32)
			).sum(axis=-1)
			
			utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]
			
			# policy improvement
			policy_grid = self.best_policy(utility_grid)
			
			# trace
			utility_grids[:, :, i] = utility_grid
			policy_grids[:, :, i] = self.best_policy(utility_grid)
						
		return policy_grids, utility_grids

	# Randomly generated experience
	# this function can not be directly used by qlearning algorithm 
	# because it generates actions according to the transition matrix _T 
	# hence you need to write your own generate_experience function according to 
	# epsilon-greedy or exploration function. 
	# when you use your own function, make sure you change your rl_qlearn.py 
	# 
	# the way you use this function is by 
	#	state, reward, done = generate_experience(self._stored_state,self._stored_action)
	
	def generate_experience(self, current_state_idx, action_idx):
		sr, sc = self.grid_indices_to_coordinates(current_state_idx)
		# function flatten converts a two-dimensional array to a one-dimensional array
		# https://www.cnblogs.com/itdyb/p/5796834.html
		next_state_probs = self._T[sr, sc, action_idx, :, :].flatten()

		# choose an actions 0,1,2,3 (up, right, down, left)
		
		next_state_idx = np.random.choice(np.arange(next_state_probs.size),p=next_state_probs)

		return (next_state_idx,
			 self._reward_grid.flatten()[next_state_idx],
			 self._terminal_mask.flatten()[next_state_idx])

	# Converts a flat index or array of flat indices into a tuple of coordinate arrays
	# https://blog.csdn.net/dn_mug/article/details/70256109
	def grid_indices_to_coordinates(self, indices=None):
		if indices is None:
			indices = np.arange(self.size)
		return np.unravel_index(indices, self.shape)

	# Converts a tuple of coordinate arrays indices into a flat index or array of flat
	# https://blog.csdn.net/Laox1ao/article/details/73289320
	def grid_coordinates_to_indices(self, coordinates=None):
		# Annoyingly, this doesn't work for negative indices.
		# The mode='wrap' parameter only works on positive indices.
		if coordinates is None:
			return np.arange(self.size)
		return np.ravel_multi_index(coordinates, self.shape)

	# Look for the best policy
	def best_policy(self, utility_grid):
		M, N = self.shape
		return np.argmax((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
					.sum(axis=-1).sum(axis=-1), axis=2)

	# create initial policy and value(utility) matrix with all zeros
	# the size of these two matrixes are MxNx iterationDepth
	def _init_utility_policy_storage(self, depth):
		M, N = self.shape
		utility_grids = np.zeros((M, N, depth))
		policy_grids = np.zeros_like(utility_grids)
		return utility_grids, policy_grids

	# This function create transition matrix with the size of MxNxnum_actionsxMxN
	# _T[0,0,1,:,:] is
	#
	# [[0.1 0.8 0.	0.	0.	0.	0.	0. ]
	#	[0.1 0.	0.	0.	0.	0.	0.	0. ]
	#	[0.	0.	0.	0.	0.	0.	0.	0. ]
	#	[0.	0.	0.	0.	0.	0.	0.	0. ]
	#	[0.	0.	0.	0.	0.	0.	0.	0. ]
	#	[0.	0.	0.	0.	0.	0.	0.	0. ]]
	#
	# It means in position (0,0), choose action 1 (right)
	# Possibility 0.1 stays in place, possibility 0.8 moves to the right and possibility 0.1 moves down
	def _create_transition_matrix(self,
							action_probabilities,
							no_action_probability,
							obstacle_mask):
		M, N = self.shape

		T = np.zeros((M, N, self._num_actions, M, N))

		r0, c0 = self.grid_indices_to_coordinates()

		T[r0, c0, :, r0, c0] += no_action_probability

		for action in range(self._num_actions):
			for offset, P in action_probabilities:
			 direction = (action + offset) % self._num_actions

			 dr, dc = self._direction_deltas[direction]
			 r1 = np.clip(r0 + dr, 0, M - 1)
			 c1 = np.clip(c0 + dc, 0, N - 1)

			 temp_mask = obstacle_mask[r1, c1].flatten()
			 r1[temp_mask] = r0[temp_mask]
			 c1[temp_mask] = c0[temp_mask]

			 T[r0, c0, action, r1, c1] += P

		terminal_locs = np.where(self._terminal_mask.flatten())[0]
		T[r0[terminal_locs], c0[terminal_locs], :, :, :] = 0
		return T

	# This function computes one iteration of the value iteration for all 
	# state s in the grid, compute vk+1(s) <- f(vk(s))
	def _value_iteration(self, utility_grid, discount=1.0):
		out = np.zeros_like(utility_grid)
		M, N = self.shape
		for i in range(M):
			for j in range(N):
			 out[i, j] = self._calculate_utility((i, j),
											discount,
											utility_grid)
		return out

	#	This function computes the v_k+1(s) <- max_a T(s,a,s')*gamma*vk(s) + R(s)
	#	Because R(s) is not related to a and s', hence we can add it as the last
	#	term in the equation.
	#	If you find numpy max and sum is very challenging to understand, you can 
	#	consider the following code which computes the same thing for the returned 
	#	v_k+1(loc)

	#	M, N = self.shape
	#	num_actions = self._num_actions
	#	max_utility = 0
	#	for current_action in range(num_actions):
	#		current_vs = np.sum(np.sum(self._T[row, col, current_action, :, :] * utility_grid,
	#				axis=-1),axis=-1) 
	#		updated_util = discount * current_vs + self._reward_grid[loc]
	#		if updated_util > max_utility:
	#		 max_utility = updated_util

	def _calculate_utility(self, loc, discount, utility_grid):
		if self._terminal_mask[loc]:
			return self._reward_grid[loc]
		row, col = loc
		return np.max(
			discount * np.sum(
			 np.sum(self._T[row, col, :, :, :] * utility_grid,
					axis=-1),
			 axis=-1)
		) + self._reward_grid[loc]

	def plot_policy(self, utility_grid, policy_grid=None):
		if policy_grid is None:
			policy_grid = self.best_policy(utility_grid)
		markers = "^>v<"
		marker_size = 200 // np.max(policy_grid.shape)
		marker_edge_width = marker_size // 10
		marker_fill_color = 'w'

		no_action_mask = self._terminal_mask | self._obstacle_mask

		utility_normalized = (utility_grid - utility_grid.min()) / \
						(utility_grid.max() - utility_grid.min())

		utility_normalized = (255*utility_normalized).astype(np.uint8)

		utility_rgb = cv2.applyColorMap(utility_normalized, cv2.COLORMAP_JET)
		for i in range(3):
			channel = utility_rgb[:, :, i]
			channel[self._obstacle_mask] = 0

		plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')

		for i, marker in enumerate(markers):
			y, x = np.where((policy_grid == i) & np.logical_not(no_action_mask))
			plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
				 color=marker_fill_color)

		y, x = np.where(self._terminal_mask)
		plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
				color=marker_fill_color)

		tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
		tick_step = np.max(policy_grid.shape)/8
		best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
		tick_step = tick_step_options[best_option]
		plt.xticks(np.arange(0, policy_grid.shape[1] - 0.5, tick_step))
		plt.yticks(np.arange(0, policy_grid.shape[0] - 0.5, tick_step))
		plt.xlim([-0.5, policy_grid.shape[0]-0.5])
		plt.xlim([-0.5, policy_grid.shape[1]-0.5])
