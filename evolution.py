import os
import shutil
import neat

import simulation
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
generation = 0

MAX_NOVELTY_SCORE = 14
COST_WEIGHT = 0 # 2
DISTANCE_WEIGHT = .53
NOVELTY_WEIGHT = .96
K = 4
NUM_GENERATIONS = 200

bestest_mouse = Mouse()



def eval_genomes(genomes, config):
    global generation, bestest_mouse
    best_mouse = Mouse()

    mazes = loader.get_random_mazes(10)
    mice = {}

    # creates the population of mice, linking them to the genomes
    for genome_id, genome in genomes:
        mice[genome_id] = Mouse(mazes[0].start_cell, mazes[0].size ** 2, len(mazes))
        mice[genome_id].genome = genome

    for maze in mazes:

        # 1. try to solve the maze
        for genome_id, genome in genomes:
            mouse = mice[genome_id]
            mouse.net = neat.nn.FeedForwardNetwork.create(genome, config)

            while mouse.alive:
                inputs = mouse.get_inputs(maze)
                outputs = mouse.net.activate(inputs)
                action = outputs.index(max(outputs))
                mouse.act(action, maze)

            # 2. calculate the first fitness component (distance, cost)
            distance = maze.distance_from_goal(mouse.position)
            mouse.compute_distance_score(distance)
            mouse.compute_cost()

        # 3. after everyone's done with the maze, calculate the second fitness component (novelty)
        for genome_id, genome in genomes:
            mouse = mice[genome_id]
            others_positions = [mice[gid].position for gid, g in genomes
                                if gid != genome_id]
            mouse.compute_novelty_score(others_positions, MAX_NOVELTY_SCORE, K)

    # 4. after all mazes, compute the final fitness
    for genome_id, genome in genomes:
        mouse = mice[genome_id]
        mouse.compute_fitness_score(DISTANCE_WEIGHT, NOVELTY_WEIGHT, COST_WEIGHT)
        genome.fitness = mouse.fitness

        # update the best individual of all
        if genome.fitness > best_mouse.fitness:
            best_mouse = mice[genome_id]
            if genome.fitness > bestest_mouse.fitness:
                bestest_mouse = mice[genome_id]

    # start the simulation every 10 generations
    if generation % 10 == 0:
        print(f"\n \n \n Simulation of the best mouse of generation {generation}... \n \n \n")
        try:
            simulation.run(best_mouse.net, generation)
        except Exception as e:
            print(e)
    generation += 1

def run(config_file):
    directory = "nets"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, None, os.path.join(directory, 'neat-checkpoint-')))


    winner = p.run(eval_genomes, NUM_GENERATIONS)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    path = os.path.join(directory, "winner_genome.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(winner, f)

    print(f"\n✅ Saved the final mouse to {path}")

    print(f"\n \n \n Simulation of the BESTEST mouse... \n \n \n")
    simulation.run(bestest_mouse.net)

    return winner, net, stats

def debug():
    global mices
    filename = "log.txt"
    with open(filename, 'w') as f:
        f.write("EVOLUTION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"\nALL MICE:\n")
        f.write("-" * 80 + "\n\n")
        for gid in mices:
            mouse = mices[gid]
            maze = loader.get_random_maze()
            fitness = f"fitness: {sum(mouse.fitness_values)/10} = {DISTANCE_WEIGHT} * {sum(mouse.distance_scores_values)/10} + {NOVELTY_WEIGHT} * {sum(mouse.novelty_scores_values)/10} - {COST_WEIGHT} * {sum(mouse.costs)/10}\n"
            position = f"last position: {mouse.position} -> {maze.distance_from_goal(mouse.position)} from goal\n"
            status = f"arrived? {mouse.arrived}. stuck? {mouse.stuck}. \n"
            f.write(fitness)
            f.write(position)
            f.write(status)
            f.write("-" * 80 + "\n")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.ini')
    run(config_path)

