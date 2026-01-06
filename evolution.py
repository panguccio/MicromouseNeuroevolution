import glob
import os
import pickle
import random
import neat
import visualize
import simulation
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
generation = 0

K = 4
NUM_GENERATIONS = 200
max_checkpoints = 3
n_mazes = 5
checkpoint_interval = 100
maze_load_interval = 50
simulate = True

bestest_mouse = Mouse()
bestest_path = os.path.join("nets", "bestest_mouse.pkl")

directory = "nets"

# for debug
mices = {}
mazes = loader.get_random_mazes(n_mazes)
counter = 0

def load_new_mazes():
    global mazes
    mazes = loader.get_random_mazes(n_mazes)
    print("🐁 Loaded new mazes:")
    for maze in mazes:
        print(f"\t * {maze.name}")
    print("\n")

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

        # try to solve the maze
        for genome_id, genome in genomes:
            mouse = mice[genome_id]
            mouse.reset()
            mouse.net = neat.nn.RecurrentNetwork.create(genome, config)

            while mouse.alive:
                inputs = mouse.get_inputs(maze)
                outputs = mouse.net.activate(inputs)
                action = outputs.index(max(outputs))
                mouse.act(action, maze)

            # compute fitness for the single maze
            mouse.compute_maze_score(maze)

    # DEBUG
    fitness_values = []

    # after all mazes, compute the final fitness
    for genome_id, genome in genomes:
        mouse = mice[genome_id]
        mouse.compute_fitness_score()
        genome.fitness = mouse.fitness
        fitness_values.append(mouse.fitness)

        # update the best individual of all
        if generation % checkpoint_interval == 0 and genome.fitness > best_mouse.fitness:
            best_mouse = mice[genome_id]
            mices[genome_id] = best_mouse
            save_mouse(best_mouse, "latest")
        if genome.fitness > bestest_mouse.fitness:
            bestest_mouse = mice[genome_id]
            save_mouse(bestest_mouse, "bestest")

            # Debug: Print some fitness values
        if len(fitness_values) <= 5:
            print(f"Genome {genome_id}: fitness = {genome.fitness}")

    # Check fitness distribution
    print(f"Fitness range: {min(fitness_values):.2f} to {max(fitness_values):.2f}")
    print(f"Fitness average: {sum(fitness_values) / len(fitness_values):.2f}")


    if generation % maze_load_interval == 0:
        counter += 1
        if best_mouse.fitness >= 200 or counter > 4: # if the best does pretty good on these mazes, load new ones
            counter = 0
            load_new_mazes()

    # start the simulation every x generations
    if generation % checkpoint_interval == 0 and best_mouse is not None and simulate:
        print(f"🎬 Simulation of the best mouse of generation {generation}... \n")
        simulation.run(best_mouse, mazes[random.randint(0, n_mazes - 1)], config)
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

        # if there's a checkpoint available, it resumes the evolution from there
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
    p.add_reporter(neat.Checkpointer(checkpoint_interval, None, os.path.join(directory, 'neat-checkpoint-')))

    p.run(eval_genomes, NUM_GENERATIONS)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    debug()

    if simulate:
        print(f"🎬 Simulation of the BESTEST mouse... \n")
        simulation.run(sim_mouse=bestest_mouse, maze=mazes[random.choice(range(n_mazes))], config=config)

def save_mouse(mouse, name):
    path = os.path.join(directory, str(name) + "_mouse.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(mouse, f)
    print(f"✅ Saved the {name} mouse to {path}")
    print(mouse_stats(mouse))
    return path

def debug():
    global mices
    filename = "log.txt"
    with open(os.path.join(directory, filename), 'a') as f:
        f.write("\n" + "=" * 80 + "\n\n")
        for gid in mices:
            mouse = mices[gid]
            f.write(mouse_stats(mouse))
            f.write("-" * 80 + "\n")

def mouse_stats(mouse):
    genetics = f"\tgeneration: {mouse.generation}; gid: {mouse.gid}\n"
    position = f"\tlast position: {mouse.position} -> {mazes[0].man_distance_from_goal(mouse.position)} from goal\n"
    fitness = f"\tfitness: {mouse.fitness} = {mouse.fitness_values}\n"
    status = f"\tarrived? {mouse.arrived}. stuck? {mouse.stuck}. \n"
    path = f"\tsteps: {mouse.actions}, num visited: {len(mouse.visited_cells)}, visited rate: {mouse.visit_rate_cost()}\n"
    return genetics + status + position + fitness + path

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.ini')
    run(config_path)


