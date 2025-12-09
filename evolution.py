import math
import operator
import os
import shutil
import statistics
from ftplib import print_line

import neat
import visualize

import maze as mz
import simulation
from maze import Maze
from maze_loader import MazeLoader
from mouse import Mouse

loader = MazeLoader()
generation = 1

# AGGIUNTO: Struttura per salvare statistiche
evolution_stats = {
    'generations': [],
    'novelty_scores': [],
    'final_positions': [],
    'best_fitness': [],
    'avg_fitness': []
}
MAX_NOVELTY_SCORE = 14
COST_WEIGHT = 0
DISTANCE_WEIGHT = 1#53
NOVELTY_WEIGHT = 1#96

def compute_novelty_score(genome_id, genome_final_positions, k=15):
    # [(genoma, last_position),...]
    mouse_position = next(pos for gid, pos in genome_final_positions if gid == genome_id)
    final_positions = [pair[1] for pair in genome_final_positions]

    distances = [mz.distance(mouse_position, pos) for pos in final_positions]
    k_nearest = sorted(distances)[:min(k, len(distances))]
    return sum(k_nearest) / len(k_nearest) if k_nearest else MAX_NOVELTY_SCORE

def eval_genomes(genomes, config):
    global generation
    best_genome = None
    best_fitness = float('-inf')

    mazes = [loader.get_random_maze()]

    # Per statistiche della generazione
    gen_novelty_scores = []
    gen_final_positions = []
    gen_fitnesses = []


    for maze in mazes:
        genome_final_positions = []  # Reset per ogni labirinto

        genome_count = 0
        for genome_id, genome in genomes:
            genome_count += 1
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mouse = Mouse(maze.start_cell, maze.size ** 2 * 3)

            while mouse.alive:

                try:
                    inputs = mouse.get_inputs(maze)
                    outputs = net.activate(inputs)
                    action = outputs.index(max(outputs))
                    mouse.act(action, maze)
                except Exception as e:
                    print(f"❌ Error during activation: {e}")
                    print(f"Inputs: {inputs}")
                    mouse.alive = False
                    break

            genome_final_positions.append((genome_id, mouse.last_position))
            gen_final_positions.append(mouse.last_position)

            # Calcola la prima fitness
            fitness = mouse.compute_distance_fitness(maze, COST_WEIGHT)

            genome.fitness = fitness

        for i in genome_final_positions:
            print(i)

        for genome_id, genome in genomes:

            distance_score = genome.fitness
            novelty_score = compute_novelty_score(genome_id, genome_final_positions)
            gen_novelty_scores.append(novelty_score)

            fitness = DISTANCE_WEIGHT * distance_score + NOVELTY_WEIGHT * novelty_score
            genome.fitness = max(fitness, 0)
            gen_fitnesses.append(genome.fitness)

    # Media fitness su tutti i labirinti
    for genome_id, genome in genomes:
        genome.fitness = genome.fitness / len(mazes)

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

    # Salva statistiche della generazione
    avg_fitness = statistics.mean(gen_fitnesses) if gen_fitnesses else 0
    evolution_stats['generations'].append({
        'generation': generation,
        'novelty_scores': gen_novelty_scores.copy(),
        'final_positions': gen_final_positions.copy(),
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness,
        'num_genomes': len(genomes)
    })

    # Stampa statistiche
    print_generation_stats(generation, gen_novelty_scores, gen_final_positions, gen_fitnesses)

    # Visualizza solo il migliore ogni 5 generazioni
    if best_genome and generation % 5 == 0:
        print(f"\n🎬 Visualizing best genome of generation {generation}...")
        simulation.run(mazes, best_genome, best_fitness, generation, config)

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

    print("🚀 Starting evolution...")
    winner = p.run(eval_genomes, 100)

    print("\n" + "=" * 80)
    print("🏆 EVOLUTION COMPLETED!")
    print("=" * 80)
    print(f"\nBest Genome:")
    print(winner)

    # Salva statistiche complete
    save_stats_to_file()

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    path = os.path.join(directory, "winner_genome.pkl")
    with open(path, "wb") as f:
        import pickle
        pickle.dump(winner, f)

    print(f"\n✅ Saved best genome to {path}")

    return winner, net, stats


def print_generation_stats(gen, novelty_scores, final_positions, fitnesses):
    """Stampa statistiche leggibili per la generazione"""
    print("\n" + "=" * 80)
    print(f"📊 GENERATION {gen} STATISTICS")
    print("=" * 80)

    # Statistiche Novelty
    if novelty_scores:
        print(f"\n🎯 NOVELTY SCORES:")
        print(f"   Min:     {min(novelty_scores):.3f}")
        print(f"   Max:     {max(novelty_scores):.3f}")
        print(f"   Mean:    {statistics.mean(novelty_scores):.3f}")
        print(f"   Median:  {statistics.median(novelty_scores):.3f}")
        print(f"   StdDev:  {statistics.stdev(novelty_scores) if len(novelty_scores) > 1 else 0:.3f}")

    # Statistiche Posizioni Finali
    print(f"\n📍 FINAL POSITIONS ({len(final_positions)} mice):")

    # Raggruppa posizioni identiche
    position_counts = {}
    for pos in final_positions:
        position_counts[pos] = position_counts.get(pos, 0) + 1

    # Mostra le top 5 posizioni più frequenti
    sorted_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 5 most common positions:")
    for i, (pos, count) in enumerate(sorted_positions[:5], 1):
        percentage = (count / len(final_positions)) * 100
        print(f"   {i}. {pos}: {count} mice ({percentage:.1f}%)")

    # Diversità posizioni
    unique_positions = len(position_counts)
    diversity = (unique_positions / len(final_positions)) * 100
    print(f"\n   Unique positions: {unique_positions}/{len(final_positions)} ({diversity:.1f}% diversity)")

    # Statistiche Fitness
    if fitnesses:
        print(f"\n💪 FITNESS:")
        print(f"   Min:     {min(fitnesses):.2f}")
        print(f"   Max:     {max(fitnesses):.2f}")
        print(f"   Mean:    {statistics.mean(fitnesses):.2f}")
        print(f"   Median:  {statistics.median(fitnesses):.2f}")

    print("=" * 80 + "\n")


def save_stats_to_file(filename="evolution_stats.txt"):
    """Salva tutte le statistiche in un file"""
    with open(filename, 'w') as f:
        f.write("EVOLUTION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        for i, gen_data in enumerate(evolution_stats['generations']):
            f.write(f"\nGENERATION {gen_data['generation']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Fitness: {gen_data['best_fitness']:.2f}\n")
            f.write(f"Avg Fitness:  {gen_data['avg_fitness']:.2f}\n")
            f.write(f"\nNovelty Scores ---- Final Positions\n")
            for index in range(len(gen_data['final_positions'])):
                f.write(f"{gen_data['novelty_scores'][index]} --- {gen_data['final_positions'][index]}\n")
            f.write("\n" + "=" * 80 + "\n")

    print(f"✅ Statistics saved to {filename}")



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.ini')
    run(config_path)

