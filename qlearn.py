import numpy as np
import random

class QLearner:

	def __init__(self, *,
				 num_states,
				 num_actions,
				 # learning_rate = alpha
				 learning_rate,
				 # discount_rate = gamma
				 discount_rate=1.0,
				 # random_action_prob = epsilon
				 random_action_prob=0.5,
				 random_action_decay_rate=0.99,
				 dyna_iterations=0):

		self._num_states = num_states
		self._num_actions = num_actions
		self._learning_rate = learning_rate
		self._discount_rate = discount_rate
		self._random_action_prob = random_action_prob
		self._random_action_decay_rate = random_action_decay_rate
		self._dyna_iterations = dyna_iterations

		self._experiences = []

		# Initialize Q table to small random values.
		self._Q = np.zeros((num_states, num_actions), dtype=np.float)
		# The value in the Q value matrix is subject to the gaussian distribution with mean of 0 and standard deviation of 0.3
		self._Q += np.random.normal(0, 0.3, self._Q.shape)

	def learn(self, initial_state, experience_func, iterations=100):
		'''Iteratively experience new states and rewards'''
		# use self._Q as Q table
		# function learn use
		#
		# policy = np.argmax(self._Q, axis=1)
		# utility = np.max(self._Q, axis=1)
		#
		# as return value
		# the reason we store policy changes and utility 
		# changes it to plot out the learning progress in the 
		# main function
		all_policies = np.zeros((self._num_states, iterations))
		all_utilities = np.zeros_like(all_policies)

		explore_on = True
		
		restart = False
		
		state = initial_state
		action = None
		
		for i in range(iterations):
			explore_rate = self._random_action_decay_rate**i* self._random_action_prob
			for j in range(300):
				if restart:
					state = initial_state
					restart = False
				
				# determine explore or not
				explore_flag = False
				if explore_on:
					if random.random() < explore_rate:
						explore_flag = True
				
				if explore_flag:
					# Explore
					action = int(random.random()*4)
				else:
					action = np.argmax(self._Q[state,:])
					
				next_state, reward, terminal_flag = experience_func(state,action)
				
				# renew Q table
				# Q[s,a] = (1-lr)*Q[s,a] + lr*(r + y*np.max(Q[s1,:])
				

				self._Q[state][action] = (1 - self._learning_rate)*self._Q[state][action] + self._learning_rate * self._discount_rate * np.max(self._Q[next_state,:])
				if terminal_flag:
					# the value of terminal should be set as the reward
					self._Q[next_state] = reward

				
				if terminal_flag:
					# if (next_state == 47):
						# print(state//8,state%8)
						# print(i)
					restart = True
					break
					
				state = next_state
			
			policy = np.argmax(self._Q, axis=1)
			utility = np.max(self._Q, axis=1)

			all_policies[:, i] = policy
			all_utilities[:, i] = utility

		return all_policies, all_utilities
	def test(self, initial_state, experience_func, iterations=100):

		score = 0
		restart = False
		state = initial_state
		action = None
		
		for i in range(iterations):
			for j in range(300):
				if restart:
					state = initial_state
					restart = False
					
				
				action = np.argmax(self._Q[state,:])
					
				next_state, reward, terminal_flag = experience_func(state,action)
				
				if terminal_flag:
					score+= reward
					restart = True
					break
					
				state = next_state

		return score