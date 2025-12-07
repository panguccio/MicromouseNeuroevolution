import os
import neat
import visualize

from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()


def compute_fitness(mouse, maze):
    distance_score = maze.distance_from_goal(mouse.position)/(maze.size - 2) # max: 14
    steps_score = mouse.steps/mouse.max_steps
    return 1 / (1 + distance_score + steps_score)


def eval_genomes(genomes, config):
    mazes = loader.get_random_mazes()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_fitness = 0
        for maze in mazes:
            mouse = Mouse(maze.start_cell, maze.size**2)

            while mouse.alive and mouse.steps < mouse.max_steps:
                inputs = mouse.get_inputs(maze)
                outputs = net.activate(inputs)
                action = outputs.index(max(outputs))
                mouse.act(action, maze)
        
            total_fitness += compute_fitness(mouse, maze)
        
        genome.fitness = total_fitness / len(mazes)


def run(config_file):
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
    p.add_reporter(neat.Checkpointer(10))  # salva ogni 10 gen

    # Run evoluzione
    winner = p.run(eval_genomes, 200)

    print("\n=== Best Genome ===")
    print(winner)

    # Costruisco la rete finale
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Salvataggio finale
    with open("winner_genome.pkl", "wb") as f:
        import pickle
        pickle.dump(winner, f)

    print("\nSaved best genome to winner_genome.pkl")

    return winner, net, stats


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.ini')
    run(config_path)

