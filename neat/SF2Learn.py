import retro
import numpy as np
import cv2 
import neat
import pickle
import visualize


def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        input_x, input_y, input_c = env.observation_space.shape

        input_x = int(input_x/8)
        input_y = int(input_y/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        newframe = 0
        newframe2 = 0
        counter = 0
        player1health_max = 176
        player2health_max = 176
        
        done = False

        while not done:
            
            env.render()
            frame += 1

            #Vektorialisieren der Bilder für den Input 
            ob = cv2.resize(ob, (input_x, input_y))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (input_x,input_y))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)
            
            # Informationen über Anzahl der Siege und Lebenspunkteanzahl beider Charaktere 
            player1health = info['health']
            player2health = info['enemy_health']
            player1wins = info['matches_won']
            player2wins = info['player2wins']
            
            #Abfragen wann Fitness dazu gewonnen wird. 
            if player2health < player2health_max:
                fitness_current += 100
                player2health_max = player2health

            if player2health == 0 and (newframe2 == frame or newframe2 == 0):
                fitness_current += 500
                newframe2 = frame

            if player1wins == 1:
                counter = 0
            if player2wins == 2:
                done = True

            if player1wins == 2 and player2wins != 2:
                counter = 0
                if frame == newframe or newframe == 0:
                    fitness_current += 100000
                    newframe = frameWeit
                    #done = True

            fitness_current += rew
            
            #Counter für Abbruchbedingung 
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            
            if done or counter == 500:
                done = True
                print(genome_id, current_max_fitness)
                
            genome.fitness = fitness_current
                
env = retro.make('StreetFighterIISpecialChampionEdition-Genesis', state='dhalsimvszangief', record='.')            
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

p = neat.Population(config)

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-3')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(3))

ob = env.reset()

sieger = p.run(eval_genomes)

visualize.draw_net(config, sieger, True)

# Erstellen einer Pickledatei für video
with open('sieg.pkl', 'wb') as output:
    pickle.dump(sieger, output, 1)

        
