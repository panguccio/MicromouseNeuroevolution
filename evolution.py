import glob
import os
import pickle
import random
import neat
import mouse
import visualize
import simulation
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
generation = 0

NUM_GENERATIONS = 600
max_checkpoints = 3
n_mazes = 1
checkpoint_interval = 100
maze_load_interval = 10000

simulate = True

bestest_mouse = Mouse()
bestest_path = os.path.join("nets", "bestest_mouse.pkl")

directory = "nets"
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
    if os.path.exists(directory) and os.listdir(directory):
        checkpoints = glob.glob(os.path.join(directory, "neat-checkpoint-*"))
        if checkpoints:
            p = restore_population(checkpoints)
        if os.path.exists(bestest_path):
            with open(bestest_path, "rb") as f:
                bestest_mouse = pickle.load(f)
    if p is None:
        os.makedirs(directory, exist_ok=True)
        p = neat.Population(config)
    return p


def restore_population(checkpoints):
    global generation
    max_index = max([int(cp.split("-")[-1]) for cp in checkpoints])
    checkpoint_file = os.path.join(directory, 'neat-checkpoint-' + str(max_index))
    p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    generation = p.generation
    for old_file in checkpoints:
        if not old_file == checkpoint_file:
            os.remove(old_file)
    print(f"Deleted old checkpoints")
    return p

def load_new_mazes(best_mouse):
    global counter, mazes
    if generation % maze_load_interval == 0:
        counter += 1
        if best_mouse.fitness >= mouse.BONUS or counter == 3:  # if the best does pretty good on these mazes, load new ones
            counter = 0
            mazes = loader.get_random_mazes(n_mazes)
            print("🐁 Loaded new mazes:")
            for maze in mazes:
                print(f"\t * {maze.name}")
            print("\n")

def start_simulation(best_mouse, config):
    if not simulate:
        return
    if generation % checkpoint_interval == 0 and best_mouse is not None and simulate:
        print(f"🎬 Simulation of the best mouse of generation {generation}... \n")
        simulation.run(best_mouse, mazes[random.randint(0, n_mazes - 1)], config)

def update_best(best_mouse, genome, genome_id, mice):
    global bestest_mouse
    if generation % checkpoint_interval == 0 and genome.fitness > best_mouse.fitness:
        best_mouse = mice[genome_id]
        save_mouse(best_mouse, "latest")
    if genome.fitness > bestest_mouse.fitness:
        bestest_mouse = mice[genome_id]
        save_mouse(bestest_mouse, "bestest")
    return best_mouse

def save_mouse(m, name):
    path = os.path.join(directory, str(name) + "_mouse.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(m, f)
    print(f"✅ Saved the {name} mouse to {path}")
    print(m.stats())
    return path


def debug():
    global mices
    print("saving the best mice")
    filename = "log.txt"
    with open(os.path.join(directory, filename), 'a') as f:
        f.write("\n" + "=" * 80 + "\n\n")
        for gid in mices:
            m = mices[gid]
            f.write(m.stats())
            f.write("-" * 80 + "\n")


def eval_genomes(genomes, config):
    global generation, bestest_mouse, mices, counter
    best_mouse = Mouse()

    mice = {}

    # creates the population of mice, linking them to the genomes
    for genome_id, genome in genomes:
        mice[genome_id] = Mouse(start_position=mazes[0].start_cell,
                                genome=genome,
                                gid=genome_id,
                                generation=generation)

    for maze in mazes:
        for genome_id, genome in genomes:
            mice[genome_id].explore(config, maze)

    for genome_id, genome in genomes:
        mice[genome_id].compute_fitness_score()
        best_mouse = update_best(best_mouse, genome, genome_id, mice)
        mices[genome_id] = best_mouse

    load_new_mazes(best_mouse)
    start_simulation(best_mouse, config)
    generation += 1


def run():
    global generation
    global directory, bestest_mouse
    p = configure_population()
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(checkpoint_interval, None, os.path.join(directory, 'neat-checkpoint-')))

    p.run(eval_genomes, NUM_GENERATIONS)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    debug()

    start_simulation(bestest_mouse, config)
    if simulate:
        print(f"🎬 Simulation of the BESTEST mouse... \n")
        simulation.run(sim_mouse=bestest_mouse, maze=mazes[random.choice(range(n_mazes))], config=config)


if __name__ == '__main__':
    run()
