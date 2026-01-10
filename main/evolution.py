import glob
import os
import pickle
import random

import neat

import visualize
from main import simulation, maze as mz
from main.mouse import Mouse
from maze_loader import MazeLoader

loader = MazeLoader()
generation = 0

NUM_GENERATIONS = 10
max_checkpoints = 3
n_mazes = 1
checkpoint_interval = 100
maze_load_interval = 100

simulate = True

bestest_mouse = None
bestest_path = os.path.join("./nets", "bestest_mouse.pkl")

nets_directory = "./nets"
config_path = os.path.join(os.path.dirname(__file__), 'config-neat.ini')
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# for debug
mices = {}
mazes = loader.get_random_mazes(n_mazes)
counter = 0


def configure_population():
    global bestest_mouse
    p = None
    if os.path.exists(nets_directory) and os.listdir(nets_directory):
        print(os.path)
        print(os.listdir(nets_directory))
        checkpoints = glob.glob(os.path.join(nets_directory, "neat-checkpoint-*"))
        if checkpoints:
            p = restore_population(checkpoints)
        if os.path.exists(bestest_path):
            with open(bestest_path, "rb") as f:
                bestest_mouse = pickle.load(f)
    if p is None:
        os.makedirs(nets_directory, exist_ok=True)
        p = neat.Population(config)
    return p


def restore_population(checkpoints):
    global generation
    max_index = max([int(cp.split("-")[-1]) for cp in checkpoints])
    checkpoint_file = os.path.join(nets_directory, 'neat-checkpoint-' + str(max_index))
    p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    generation = p.generation
    for old_file in checkpoints:
        if not old_file == checkpoint_file:
            os.remove(old_file)
    print(f"Deleted old checkpoints")
    return p


def load_new_mazes():
    global counter, mazes
    if (generation - 1) % maze_load_interval == 0:
        mazes = loader.get_random_mazes(n_mazes)
        print("---> Loaded new mazes:")
        for maze in mazes:
            print(f"\t * {maze.name}")
        print("\n")


def start_simulation(best_mouse):
    if not simulate:
        return
    if generation % checkpoint_interval == 0 and best_mouse is not None and simulate:
        print(f"--> Simulation of the best mouse of generation {generation}... \n")
        simulation.run(best_mouse, mazes[random.randint(0, n_mazes - 1)], config)


def update_bestest(best_mouse):
    global bestest_mouse, mices

    if generation % checkpoint_interval == 0 and best_mouse is not None:
        save_mouse(best_mouse, "latest")

    if bestest_mouse is None or best_mouse.genome.fitness > bestest_mouse.genome.fitness:
        bestest_mouse = best_mouse
        mices[bestest_mouse.gid] = bestest_mouse
        save_mouse(bestest_mouse, "bestest")

    return best_mouse


def save_mouse(m, name):
    path = os.path.join(nets_directory, str(name) + "_mouse.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(m, f)
    print(f"---> Saved the {name} mouse to {path}")
    print(m.stats())
    return path


def debug():
    global mices
    print("saving the best mice")
    filename = "log.txt"
    with open(os.path.join(nets_directory, filename), 'a') as f:
        f.write("\n" + "=" * 80 + "\n\n")
        for gid in mices:
            m = mices[gid]
            f.write(m.stats())
            f.write("-" * 80 + "\n")


def eval_genomes(genomes, _):
    global generation, mice
    best_mouse = None

    mice = {}

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.RecurrentNetwork.create(genome, config)
        mice[genome_id] = Mouse(start_position=mz.start_cell,
                                genome=genome,
                                gid=genome_id,
                                generation=generation,
                                net=net)

    for maze in mazes:
        for mouse in mice.values():
            mouse.explore(maze)

    for genome_id, genome in genomes:
        if best_mouse is None or genome.fitness > best_mouse.genome.fitness:
            best_mouse = mice[genome_id]
        genome.fitness = genome.fitness / len(mazes)

    update_bestest(best_mouse)
    load_new_mazes()
    start_simulation(best_mouse)
    generation += 1


def run():
    global generation
    global nets_directory, bestest_mouse
    p = configure_population()
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(checkpoint_interval, None, os.path.join(nets_directory, 'neat-checkpoint-')))

    p.run(eval_genomes, NUM_GENERATIONS)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join('./images', 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join('./images', 'avg_fitness.svg'))
    debug()

    start_simulation(bestest_mouse)


if __name__ == '__main__':
    print(os.path.dirname(os.path.realpath(__file__)))
    run()
