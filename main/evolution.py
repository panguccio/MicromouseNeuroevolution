import glob
import os
import pickle
import random

import neat

import visualize
from main import simulation, maze as mz
from main.mouse import Mouse
from maze_loader import MazeLoader


class NEATTrainer:
    """Handles the training of the NEAT population to solve labyrinths."""

    def __init__(self, config_path='config-neat.ini'):
        self.loader = MazeLoader()
        self.generation = 0

        # Config
        self.NUM_GENERATIONS = 1000
        self.MAX_CHECKPOINTS = 3
        self.N_MAZES = 1
        self.CHECKPOINT_INTERVAL = 50
        self.MAZE_LOAD_INTERVAL = 150
        self.SIMULATE = True

        # Paths
        self.nets_directory = "./nets"
        self.bestest_path = os.path.join(self.nets_directory, "bestest_mouse.pkl")
        self.images_directory = "./images"

        # State
        self.bestest_mouse = None
        self.best_mice = {}
        self.mazes = self.loader.get_random_mazes(self.N_MAZES)

        # NEAT config
        full_config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            full_config_path
        )

    # ---
    # Memory
    # ---

    def configure_population(self):
        """Configures the NEAT population, if it can it restores the session from a checkpoint."""
        p = None

        if os.path.exists(self.nets_directory) and os.listdir(self.nets_directory):
            checkpoints = glob.glob(os.path.join(self.nets_directory, "neat-checkpoint-*"))
            if checkpoints:
                p = self._restore_population(checkpoints)

            if os.path.exists(self.bestest_path):
                with open(self.bestest_path, "rb") as f:
                    self.bestest_mouse = pickle.load(f)

        if p is None:
            os.makedirs(self.nets_directory, exist_ok=True)
            p = neat.Population(self.config)

        return p

    def _restore_population(self, checkpoints):
        """Restores the population from the most recent checkpoint."""
        max_index = max(int(cp.split("-")[-1]) for cp in checkpoints)
        checkpoint_file = os.path.join(self.nets_directory, f'neat-checkpoint-{max_index}')

        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        self.generation = p.generation

        # Elimina i checkpoint vecchi
        for old_file in checkpoints:
            if old_file != checkpoint_file:
                os.remove(old_file)

        print(f"- Restored population of gen: {self.generation}")
        print("- Deleted old checkpoints")

        return p

    def load_new_mazes(self):
        """Loads new mazes."""
        if (self.generation - 1) % self.MAZE_LOAD_INTERVAL == 0:
            self.mazes = self.loader.get_random_mazes(self.N_MAZES)
            print("\n-> New mazes loaded:")
            for maze in self.mazes:
                print(f"     * {maze.name}")
            print()

    def simulate_best_mouse(self, best_mouse):
        """Executes the simulation of the best mouse."""
        if not self.SIMULATE:
            return

        if self.generation % self.CHECKPOINT_INTERVAL == 0 and best_mouse is not None:
            print(f"\n--> Simulation of the best mouse (gen: {self.generation})...\n")
            random_maze = random.choice(self.mazes)
            simulation.run(best_mouse, random_maze, self.config)

    def update_bestest_mouse(self, best_mouse):
        """Updates the best mouse if necessary."""
        if self.generation % self.CHECKPOINT_INTERVAL == 0 and best_mouse is not None:
            self._save_mouse(best_mouse, "latest")

        if (self.bestest_mouse is None or
                best_mouse.genome.fitness > self.bestest_mouse.genome.fitness):
            self.bestest_mouse = best_mouse
            self.best_mice[self.bestest_mouse.gid] = self.bestest_mouse
            self._save_mouse(self.bestest_mouse, "bestest")

    def _save_mouse(self, mouse, name):
        """Saves the mouse in a file."""
        path = os.path.join(self.nets_directory, f"{name}_mouse.pkl")
        with open(path, "wb") as f:
            pickle.dump(mouse, f)

        print(f"\n-> Saved '{name}' in {path}")
        print(mouse.stats())

    def save_debug_log(self):
        """Saves mouse's stats in a log file."""
        log_path = os.path.join(self.nets_directory, "log.txt")

        with open(log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n\n")
            for gid, mouse in self.best_mice.items():
                f.write(mouse.stats())
                f.write("-" * 80 + "\n")

        print(f"\n- Log saved in {log_path}")

    # ---
    # Main methods
    # ---

    def eval_genomes(self, genomes, _):
        """Core of the evolution process."""
        best_mouse = None
        mice = {}

        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.RecurrentNetwork.create(genome, self.config)
            mice[genome_id] = Mouse(
                start_position=mz.START_CELL,
                genome=genome,
                gid=genome_id,
                generation=self.generation,
                net=net
            )

        for maze in self.mazes:
            for mouse in mice.values():
                mouse.explore(maze)

        for genome_id, genome in genomes:
            if best_mouse is None or genome.fitness > best_mouse.genome.fitness:
                best_mouse = mice[genome_id]
            genome.fitness = genome.fitness / len(self.mazes)

        self.update_bestest_mouse(best_mouse)
        self.load_new_mazes()
        self.simulate_best_mouse(best_mouse)

        self.generation += 1

    def run(self):
        """Executes the training process."""

        p = self.configure_population()
        p.add_reporter(neat.StdOutReporter(True))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        checkpointer = neat.Checkpointer(
            self.CHECKPOINT_INTERVAL,
            None,
            os.path.join(self.nets_directory, 'neat-checkpoint-')
        )
        p.add_reporter(checkpointer)

        p.run(self.eval_genomes, self.NUM_GENERATIONS)

        os.makedirs(self.images_directory, exist_ok=True)
        visualize.plot_stats(
            stats,
            ylog=False,
            view=True,
            filename=os.path.join(self.images_directory, 'avg_fitness.svg')
        )
        visualize.plot_species(
            stats,
            view=True,
            filename=os.path.join(self.images_directory, 'speciation.svg')
        )

        self.save_debug_log()
        self.simulate_best_mouse(self.bestest_mouse)

if __name__ == '__main__':
    trainer = NEATTrainer()
    trainer.run()
