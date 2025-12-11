import glob
import os
import pickle


import neat

import simulation
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
generation = 0



K = 4
NUM_GENERATIONS = 300
max_checkpoints = 3
n_mazes = 20

bestest_mouse = Mouse()
bestest_path = os.path.join("nets", "bestest_mouse.pkl")

directory = "nets"

# for debug
mices = {}


def eval_genomes(genomes, config):
    global generation, bestest_mouse, mices
    best_mouse = Mouse()

    mazes = loader.get_random_mazes(n_mazes)
    mice = {}

    # creates the population of mice, linking them to the genomes
    for genome_id, genome in genomes:
        mice[genome_id] = Mouse(start_position=mazes[0].start_cell,
                                max_steps=mazes[0].size*8,
                                genome=genome,
                                gid=genome_id,
                                generation=generation)

    for maze in mazes:

        # 1. try to solve the maze
        for genome_id, genome in genomes:
            mouse = mice[genome_id]
            mouse.reset()
            mouse.net = neat.nn.FeedForwardNetwork.create(genome, config)

            while mouse.alive:
                inputs = mouse.get_inputs(maze)
                outputs = mouse.net.activate(inputs)
                action = outputs.index(max(outputs))
                mouse.act(action, maze)

            # 2. calculate the first fitness component (distance, cost)
            distance = maze.man_distance_from_goal(mouse.position)
            mouse.compute_distance_score(distance)
            mouse.compute_cost()

        # 3. after everyone's done with the maze, calculate the second fitness component (novelty)
        for genome_id, genome in genomes:
            mouse = mice[genome_id]
            others_positions = [mice[gid].position for gid, g in genomes
                                if gid != genome_id]
            mouse.compute_novelty_score(others_positions, K)

    # 4. after all mazes, compute the final fitness
    for genome_id, genome in genomes:
        mouse = mice[genome_id]
        mouse.compute_fitness_score()
        genome.fitness = mouse.fitness

        # update the best individual of all
        if genome.fitness > best_mouse.fitness:
            best_mouse = mice[genome_id]
            save_mouse(best_mouse, "latest")
            if genome.fitness > bestest_mouse.fitness:
                bestest_mouse = mice[genome_id]
                mices[genome_id] = bestest_mouse
                save_mouse(bestest_mouse, "bestest")


    # start the simulation every 10 generations
    if generation % 100 == 0:
        print(f"🎬 Simulation of the best mouse of generation {generation}... \n")
        simulation.run(best_mouse)
    generation += 1

def run(config_file):
    global generation
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    global directory, bestest_mouse

    p = None

    if os.path.exists(directory) and os.listdir(directory):
        checkpoints = glob.glob(os.path.join(directory, "neat-checkpoint-*"))

        if checkpoints:

            max_index = max([int(cp.split("-")[-1]) for cp in checkpoints])
            checkpoint_file = os.path.join(directory, 'neat-checkpoint-' + str(max_index))
            p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            generation = p.generation

            for old_file in checkpoints:
                if not old_file == checkpoint_file:
                    os.remove(old_file)
            print(f"Deleted old checkpoints")

        if os.path.exists(bestest_path):
            with open(bestest_path, "rb") as f:
                bestest_mouse = pickle.load(f)

    if p is None:
        os.makedirs(directory, exist_ok=True)
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(49, None, os.path.join(directory, 'neat-checkpoint-')))

    p.run(eval_genomes, NUM_GENERATIONS)
    debug()

    print(f"🎬 Simulation of the BESTEST mouse... \n")
    simulation.run(bestest_mouse)


def save_mouse(mouse, name):
    path = os.path.join(directory, str(name) + "_mouse.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(mouse, f)
    print(f"✅ Saved the {name} mouse to {path}")
    print(f"\t fitness: {mouse.fitness}")
    print(f"\t gid: {mouse.gid}; generation: {mouse.generation}\n")
    return path


def debug():
    global mices
    filename = "log.txt"
    with open(os.path.join(directory, filename), 'a') as f:
        f.write("\n" + "=" * 80 + "\n\n")
        for gid in mices:
            mouse = mices[gid]
            maze = loader.get_random_maze()
            genetics = f"generation: {mouse.generation}; gid: {mouse.gid}\n"
            position = f"last position: {mouse.position} -> {maze.man_distance_from_goal(mouse.position)} from goal\n"
            fitness = f"fitness: {mouse.fitness} = {mouse.fitness_values}\n"
            distance = f"distance: {mouse.distance_scores_values}\n"
            novelty = f"novelty: {mouse.novelty_scores_values}\n"
            costs = f"costs: {mouse.costs}\n"
            status = f"arrived? {mouse.arrived}. stuck? {mouse.stuck}. \n"
            f.write(genetics + status + position + fitness + distance + novelty + costs)
            f.write("-" * 80 + "\n")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.ini')
    run(config_path)

