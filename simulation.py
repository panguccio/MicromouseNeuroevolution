import neat
import pygame
import pickle
import os

import graphics
from maze_loader import MazeLoader
from mouse import Mouse

BESTEST_PATH = os.path.join("nets", "bestest_mouse.pkl")
LATEST_PATH = os.path.join("nets", "latest_mouse.pkl")

best_simulation = False
user_controlled = False
loader = MazeLoader()

def load_best_mouse():
    path = BESTEST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Non trovo il file {path}, ne creo uno nuovo.")
    with open(path, "rb") as f:
        mouse = pickle.load(f)

    return mouse

# --- LOGIC ---
def move_with_network(maze, mouse):
    inputs = mouse.get_inputs(maze)
    outputs = mouse.net.activate(inputs)
    action = outputs.index(max(outputs))
    mouse.act(action, maze)
    mouse.update_maze_score(maze, 0)

def move_with_keys(event, maze, mouse):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            mouse.act(1, maze)
        if event.key == pygame.K_LEFT:
            mouse.act(0, maze)
        if event.key == pygame.K_RIGHT:
            mouse.act(2, maze)

def run(sim_mouse=None, maze=None, config=None):
    global best_simulation, user_controlled

    if maze is None:
        maze = loader.get_random_maze()

    if sim_mouse is None or sim_mouse.genome is None:
        try:
            sim_mouse = load_best_mouse()
        except FileNotFoundError:
            sim_mouse = Mouse()
            user_controlled = True

    mouse = Mouse(start_position=maze.start_cell,
                  generation=sim_mouse.generation,
                  fitness=sim_mouse.fitness,
                  genome=sim_mouse.genome,
                  gid=sim_mouse.gid)

    pygame.init()
    if best_simulation:
        pygame.display.set_caption(f"Micromouse Neuroevolution - Bestest mouse")
    elif user_controlled:
        pygame.display.set_caption(f"Micromouse Neuroevolution - Test")
    else:
        pygame.display.set_caption(f"Micromouse Neuroevolution - Gen {mouse.generation}")

    # Layout
    maze_pixel_size = maze.size * graphics.CELL_SIZE
    dashboard_width = 350
    screen_height = max(maze_pixel_size, 550)
    screen_width = maze_pixel_size + dashboard_width

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.event.pump()
    pygame.display.flip()
    clock = pygame.time.Clock()
    maze_offset_y = (screen_height - maze_pixel_size) // 2

    running = True

    while running:
        # Input from user
        keys = pygame.key.get_pressed()

        # Kill command
        if keys[pygame.K_k]:
            if mouse.alive:
                mouse.alive = False  # Uccidi
            elif best_simulation:
                running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            move_with_keys(event, maze, mouse)

        if mouse.net is None and not user_controlled:
            mouse.net = neat.nn.RecurrentNetwork.create(mouse.genome, config)

        # SIMULATION
        if mouse.alive:
            # Gestione Velocità: se premi S fai 20 step logici in un solo frame grafico
            steps_per_frame = 3 if keys[pygame.K_s] else 1

            for _ in range(steps_per_frame):
                if mouse.alive and not user_controlled:
                    move_with_network(maze, mouse)
                else:
                    break

        elif not best_simulation:
            running = False

        # Drawing it all!
        screen.fill(graphics.BG_COLOR)

        graphics.draw_maze(screen, mouse, maze, offset_x=0, offset_y=maze_offset_y)
        graphics.draw_mouse(screen, mouse, offset_x=0, offset_y=maze_offset_y)
        graphics.draw_dashboard(screen=screen, x=maze_pixel_size, y=0, width=dashboard_width, height=screen_height, mouse=mouse, genome=mouse.genome, maze=maze, best_simulation=best_simulation)

        pygame.display.flip()
        clock.tick(10)  # 10 FPS per la GUI

    pygame.quit()


if __name__ == '__main__':
    best_simulation = True
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-neat.ini')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    run(config=config)