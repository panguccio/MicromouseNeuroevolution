import pygame
import neat
import pickle
import os

import evolution
from maze_loader import MazeLoader
from maze import Maze
from direction import Direction
from mouse import Mouse  # oppure incolla la classe dentro lo script

CELL_SIZE = 30
BG_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)
WALL_THICKNESS = 2
MOUSE_IMG = pygame.image.load("mouse.png")
MOUSE_IMG = pygame.transform.scale(MOUSE_IMG, (CELL_SIZE // 1.5, CELL_SIZE // 1.5))

path = os.path.join("nets", "winner_genome.pkl")

def load_best_network(config_path="config-neat.ini", genome_path=path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net


def draw_maze(screen, maze: Maze):
    size = maze.size

    for r in range(size):
        for c in range(size):
            x = c * CELL_SIZE
            y = r * CELL_SIZE

            if maze.has_wall(Direction.N, r, c):
                pygame.draw.line(screen, WALL_COLOR, (x, y), (x + CELL_SIZE, y), WALL_THICKNESS)

            # Parete Sud
            if maze.has_wall(Direction.S, r, c):
                pygame.draw.line(screen, WALL_COLOR, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), WALL_THICKNESS)

            # Parete Ovest
            if maze.has_wall(Direction.W, r, c):
                pygame.draw.line(screen, WALL_COLOR, (x, y), (x, y + CELL_SIZE), WALL_THICKNESS)

            # Parete Est
            if maze.has_wall(Direction.E, r, c):
                pygame.draw.line(screen, WALL_COLOR, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), WALL_THICKNESS)


def draw_mouse(screen, mouse: Mouse):
    r, c = mouse.position
    x = c * CELL_SIZE + CELL_SIZE // 2
    y = r * CELL_SIZE + CELL_SIZE // 2

    rotated = pygame.transform.rotate(MOUSE_IMG, mouse.direction.angle)
    rect = rotated.get_rect(center=(x, y))
    screen.blit(rotated, rect)




def main():
    pygame.init()
    loader = MazeLoader()
    maze = loader.get_maze("yama7.txt")

    net = load_best_network()

    mouse = Mouse(maze.start_cell, maze.size ** 2)

    screen = pygame.display.set_mode((maze.size * CELL_SIZE, maze.size * CELL_SIZE))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # Movimento con frecce

            # move_with_keys(event, maze, mouse)
        move_with_network(maze, mouse, net)
        screen.fill(BG_COLOR)
        draw_maze(screen, maze)
        draw_mouse(screen, mouse)

        pygame.display.flip()
        clock.tick(5)

    print(mouse.compute_fitness(maze, novelty_score(mouse.position)))

    pygame.quit()


def move_with_keys(event, maze, mouse):
    if event.type == pygame.KEYDOWN:

        if event.key == pygame.K_UP:
            mouse.act(0, maze)

        if event.key == pygame.K_LEFT:
            mouse.act(1, maze)

        if event.key == pygame.K_RIGHT:
            mouse.act(2, maze)

        if event.key == pygame.K_a:
            mouse.act(3, maze)

        if event.key == pygame.K_d:
            mouse.act(4, maze)



def move_with_network(maze, mouse, net):
    if mouse.alive:
        inputs = mouse.get_inputs(maze)
        outputs = net.activate(inputs)
        action = outputs.index(max(outputs))
        mouse.act(action, maze)


if __name__ == "__main__":
    main()
