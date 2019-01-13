# Mountain Car
# state space: observation is tuple of position (-1.2 to 0.6) and velocity (-0.07 to 0.07)
# action space: 0 is push left, 1 is NOOP, 2 is push right

# An MDP is defined by:
# - a set of states S
# - a set of actions A
# - a transition function T(s,a,s'), also called the model, 
# the probability that performing action a in state s will lead to state s'
# - a reward function R(s,a,s') or just R(s), or R(s')
# the reward you get for performing an action in state S, or just for leaving state s, or arriving in state s'

# implement Q-learning with Q-table lookup

import gym
import numpy as np

MAX_NUM_EPISODES = 10000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30

class Q_Learner(object):

	def __init__(self, env):
		self.obs_shape = env.observation_space.shape
		self.obs_high = env.observation_space.high
		self.obs_low = env.observation_space.low
		self.obs_bins = NUM_DISCRETE_BINS
		self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
		self.action_shape = env.action_space.n

		# create the Q-table:
		self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,self.action_shape))
		self.alpha = ALPHA
		self.gamma = GAMMA
		self.epsilon = 1.0
		print(self.Q.shape)

	def discretize(self, obs):
		return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

	def get_action(self, obs):
		discretized_obs = self.discretize(obs)
		if self.epsilon > EPSILON_MIN:
			self.epsilon -= EPSILON_DECAY
		if np.random.random() > self.epsilon:
			return np.argmax(self.Q[discretized_obs])
		else:
			return np.random.choice([a for a in range(self.action_shape)])

	def learn(self, obs, action, reward, next_obs):
		discretized_obs = self.discretize(obs)
		discretized_next_obs = self.discretize(next_obs)
		td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
		td_error = td_target - self.Q[discretized_obs][action]
		
		self.Q[discretized_obs][action] += self.alpha * td_error

def train(agent, env):
	best_reward = -float('inf')
	for episode in range(MAX_NUM_EPISODES):
		done = False
		obs = env.reset()
		total_reward = 0.0

		while not done:
			action = agent.get_action(obs)
			next_obs, reward, done, info = env.step(action)
			agent.learn(obs, action, reward, next_obs)
			obs = next_obs
			total_reward += reward

		if total_reward > best_reward:
			best_reward = total_reward

		if episode%100 == 0:
			print("Episode#:{} reward:{} best_reward:{} eps:{}"
				.format(episode,total_reward, best_reward, agent.epsilon)
			)

	return np.argmax(agent.Q, axis=2)

def test(agent, env, policy):
	done = False
	obs = env.reset()
	total_reward = 0.0

	while not done:
		action = policy[agent.discretize(obs)]
		next_obs, reward, done, info = env.step(action)
		obs = next_obs
		total_reward += reward

	return total_reward

if __name__ == "__main__":
	env = gym.make('MountainCar-v0')
	agent = Q_Learner(env)
	print('begining training')
	learned_policy = train(agent, env)
	print('finished training')
	print(agent.Q)
	gym_monitor_path = "./mountain_car_output"

	env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
	
	print('testing policy')
	for _ in range(1000):
		test(agent, env, learned_policy)
	
	env.close()







