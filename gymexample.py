import gym
import random
import numpy as np
import tflearn
import time
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3 # Learning Rate
env = gym.make('CartPole-v0') # Construindo o ambiente
goal_steps = 800
score_requirement = 50 # Score minimo para se levado em consideracao
initial_games = 10000


def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	env.reset()
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		# Joga um jogo
		for _ in range(goal_steps):
			#env.render()
			#time.sleep(0.03)
			action = env.action_space.sample() # Gera uma ação aleatória
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation = observation

			score += reward
			if done:
				break

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]
		
				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	print 'Average accepted score:', mean(accepted_scores)
	print 'Median accepted score:', median(accepted_scores)
	print Counter(accepted_scores)

	return training_data

def neural_network_model(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8) # Manter 80% dos sinais

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8) # Manter 80% dos sinais

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8) # Manter 80% dos sinais

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8) # Manter 80% dos sinais

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8) # Manter 80% dos sinais

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):
	x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size = len(x[0]))

	model.fit({'input':x}, {'targets':y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openaistuff')

	return model


training_data = initial_population()
model = train_model(training_data)

model.save('modelo_1.model')

scores = []
choices = []

for each_game in range(5):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		time.sleep(0.02)
		if len(prev_obs) == 0:
			action = env.action_space.sample()
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation, action])
		score += reward
		if done:
			break
	scores.append(score)

print "Average Score", sum(scores)/len(scores)
choice1 = float(choices.count(1))
choice0 = float(choices.count(0))
print ("Choice 1: %f \t Choice 0: %f" % (choice1/len(choices), choice0/len(choices)))










