import retro
import numpy as np
import cv2 
import neat
import pickle
import gym

env = retro.make('StreetFighterIISpecialChampionEdition-Genesis', state='dhalsimvszangief')            
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-2')
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open('sieg.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()
input_x, input_y input_c = env.observation_space.shape

input_x = int(input_x/8)
input_y = int(input_y/8)
done = False
net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
player1health_max = 176
player2health_max = 176
player1currenthealth = 176

while not done:
	env.render()
	frame += 1
	ob = cv2.resize(ob, (input_x, input_y))
	ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
	ob = np.reshape(ob, (input_x,input_y))

	imgarray = np.ndarray.flatten(ob)

	nnOutput = net.activate(imgarray)

	ob, rew, done, info = env.step(nnOutput)

	player1health = info['health']
	player2health = info['enemy_health']
	player1wins = info['matches_won']
	player2wins = info['player2wins']

	if player2health < player2health_max:
	    fitness_current += 100
	    player2health_max = player2health

	if player2health == 0:
	    fitness_current += 500

	if player1wins == 1:
	    counter = 0
	if player2wins == 2:
		done = True

	if player1wins == 2 and player2wins != 2:
	    fitness_current += 100000
        counter = 0
	    #done = True

	fitness_current += rew

	if fitness_current > current_max_fitness:
	    current_max_fitness = fitness_current
	    counter = 0
	else:
	    counter += 1

	#if done or counter == 500:
	    #done = True
	    
	genome.fitness = fitness_current
