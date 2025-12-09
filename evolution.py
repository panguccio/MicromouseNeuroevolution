import math
import operator
import os
import shutil
import statistics

import neat
import visualize

import maze as mz
from maze import Maze
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
best_genome = None
best_fitness = float('-inf')


def novelty_score(mouse_position, final_positions, k=30):
    distances = [mz.distance(mouse_position, pos) for pos in final_positions]
    k_nearest = sorted(distances)[:k]
    final_positions.append(mouse_position)
    return sum(k_nearest) / len(k_nearest) if k_nearest else 0

def eval_genomes(genomes, config):
    global best_fitness
    global best_genome
    final_positions = []
    mazes = [loader.get_maze("yama7.txt")]
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_fitness = 0
        for maze in mazes:
            mouse = Mouse(maze.start_cell, maze.size**2)

            while mouse.alive and mouse.steps < mouse.max_steps and mouse.costs < mouse.max_steps:
                inputs = mouse.get_inputs(maze)
                outputs = net.activate(inputs)
                action = outputs.index(max(outputs))
                mouse.act(action, maze)

            total_fitness += mouse.compute_fitness(maze, novelty_score(mouse.position, final_positions, 20))
        
        genome.fitness = total_fitness / len(mazes)

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

def run(config_file):
    directory = "nets"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Population
    p = neat.Population(config)

    # Reporters (log sulla console + statistiche + checkpoint)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, None, os.path.join(directory, 'neat-checkpoint-')))  # salva ogni 10 gen

    # Run evoluzione
    winner = p.run(eval_genomes, 10)
    winner = best_genome

    print("\n=== Best Genome ===")
    print(winner)

    # Costruisco la rete finale
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Salvataggio finale
    path = os.path.join(directory, "winner_genome.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(winner, f)

    print("\nSaved best genome to winner_genome.pkl")

    return winner, net, stats


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.ini')
    run(config_path)

