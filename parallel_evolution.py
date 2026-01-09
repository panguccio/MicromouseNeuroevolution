import os
import random

import neat

import maze as mz
import simulation
import visualize
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
generation = 0

NUM_GENERATIONS = 600
n_mazes = 5
checkpoint_interval = 100
directory = "nets"

simulate = True
config_path = os.path.join(os.path.dirname(__file__), 'config-neat.ini')
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# for debug
best_mice = {}
mazes = loader.get_random_mazes(n_mazes)
counter = 0


def start_simulation(winner_genome):
    winner = mouse = Mouse(genome=winner_genome)
    if not simulate:
        return
    if generation % checkpoint_interval == 0 and winner is not None and simulate:
        print(f"🎬 Simulation of the best mouse of generation {generation}... \n")
        simulation.run(winner, mazes[random.randint(0, n_mazes - 1)], configuration=config)


def eval_genome(genome, _):
    mouse = Mouse(start_position=mz.start_cell,
                  genome=genome,
                  generation=generation)

    for maze in mazes:
        mouse.explore(config, maze)
    mouse.compute_fitness_score()
    return mouse.fitness


def run():
    global generation
    global directory, bestest_mouse
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(checkpoint_interval, None, os.path.join(directory, 'neat-checkpoint-')))

    with neat.ParallelEvaluator(4, eval_genome) as evaluator:
        winner = p.run(evaluator.evaluate, 300)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    start_simulation(winner)


if __name__ == '__main__':
    run()
