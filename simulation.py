import pygame
import neat
import pickle
import os
from maze import Maze
from direction import Direction
from mouse import Mouse

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


def draw_maze(screen, maze: Maze, offset_x=0, offset_y=0):
    size = maze.size

    for r in range(size):
        for c in range(size):
            x = c * CELL_SIZE + offset_x
            y = r * CELL_SIZE + offset_y

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


def draw_mouse(screen, mouse: Mouse, offset_x=0, offset_y=0):
    r, c = mouse.position
    x = c * CELL_SIZE + CELL_SIZE // 2 + offset_x
    y = r * CELL_SIZE + CELL_SIZE // 2 + offset_y

    rotated = pygame.transform.rotate(MOUSE_IMG, mouse.direction.angle)
    rect = rotated.get_rect(center=(x, y))
    screen.blit(rotated, rect)


def draw_stats(screen, maze: Maze, mouse: Mouse, stats_width, maze_width, screen_height, generation):
    # Pannello statistiche a destra del labirinto
    panel_x = maze_width + 20
    panel_width = stats_width - 40
    panel_height = 380

    # Centra verticalmente il pannello
    panel_y = (screen_height - panel_height) // 2

    # Sfondo del pannello (nero con bordo rosso)
    pygame.draw.rect(screen, (0, 0, 0), (panel_x - 10, panel_y - 10, panel_width + 20, panel_height), border_radius=8)
    pygame.draw.rect(screen, (255, 0, 0), (panel_x - 10, panel_y - 10, panel_width + 20, panel_height), width=3,
                     border_radius=8)

    # Font
    font_title = pygame.font.SysFont('Arial', 28, bold=True)
    font_stat = pygame.font.SysFont('Arial', 22)
    font_small = pygame.font.SysFont('Arial', 18)

    # Titolo
    title = font_title.render("STATISTICHE", True, (255, 0, 0))
    screen.blit(title, (panel_x + panel_width // 2 - title.get_width() // 2, panel_y))

    y_offset = panel_y + 50
    line_height = 35

    # Statistiche con schema colori rosso/bianco/giallo
    stats = [
        ("Generation:", str(generation), (255, 255, 0)),
        ("Steps:", str(mouse.steps), (255, 255, 255)),
        ("Alive:", str(mouse.alive), (255, 255, 0)),
        ("Fitness:", f"{mouse.fitness:.2f}", (255, 100, 100)),
    ]

    for label, value, color in stats:
        # Label in grigio chiaro
        label_surf = font_stat.render(label, True, (180, 180, 180))
        screen.blit(label_surf, (panel_x, y_offset))

        # Valore con colore
        value_surf = font_stat.render(value, True, color)
        screen.blit(value_surf, (panel_x + 130, y_offset))

        y_offset += line_height

    # Separatore rosso
    y_offset += 10
    pygame.draw.line(screen, (255, 0, 0), (panel_x, y_offset), (panel_x + panel_width, y_offset), 2)
    y_offset += 20

    # Action
    action_label = font_stat.render("Action:", True, (180, 180, 180))
    screen.blit(action_label, (panel_x, y_offset))
    y_offset += 30

    action_value = font_stat.render(str(mouse.last_action), True, (255, 255, 0))
    screen.blit(action_value, (panel_x + 20, y_offset))
    y_offset += 45

    # Inputs (formattati meglio)
    inputs_label = font_stat.render("Inputs:", True, (180, 180, 180))
    screen.blit(inputs_label, (panel_x, y_offset))
    y_offset += 30

    if mouse.last_inputs:
        # Mostra gli inputs su più righe se necessario
        inputs_str = str([f"{x}" for x in mouse.last_inputs])
        # Dividi in righe se troppo lungo
        max_chars = 100
        if len(inputs_str) > max_chars:
            # Mostra solo i primi valori
            inputs_display = f"[{len(mouse.last_inputs)} valori]"
        else:
            inputs_display = inputs_str

        inputs_surf = font_small.render(inputs_display, True, (200, 200, 200))
        screen.blit(inputs_surf, (panel_x + 10, y_offset))


def run(mazes, best_genome, best_fitness, generation, config):
    pygame.init()

    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    for maze in mazes:
        mouse = Mouse(maze.start_cell, maze.size ** 2)
        mouse.fitness = best_fitness

        # Calcola dimensioni finestra
        maze_width = maze.size * CELL_SIZE
        maze_height = maze.size * CELL_SIZE
        stats_width = 300  # Larghezza pannello statistiche

        screen_width = maze_width + stats_width + 20
        screen_height = max(maze_height, 450)  # Altezza minima per statistiche

        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Maze AI - Generation {generation}")

        # Offset per centrare il labirinto verticalmente
        maze_offset_y = (screen_height - maze_height) // 2

        clock = pygame.time.Clock()

        while mouse.alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    mouse.alive = False
                # Movimento con frecce (opzionale)
                # move_with_keys(event, maze, mouse)

            move_with_network(maze, mouse, net)

            # Disegna tutto
            screen.fill(BG_COLOR)
            draw_maze(screen, maze, offset_x=10, offset_y=maze_offset_y)
            draw_mouse(screen, mouse, offset_x=10, offset_y=maze_offset_y)
            draw_stats(screen, maze, mouse, stats_width, maze_width, screen_height, generation)

            pygame.display.flip()
            clock.tick(5)

        clock.tick(10)
        print(f"Mouse fitness in generation {generation}: {best_fitness}")


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
    inputs = mouse.get_inputs(maze)
    outputs = net.activate(inputs)
    action = outputs.index(max(outputs))

    mouse.last_inputs = inputs
    mouse.last_action = action

    mouse.act(action, maze)