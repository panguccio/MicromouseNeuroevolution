import os
import pickle

import neat
import pygame

import graphics
from maze_loader import MazeLoader
from mouse import Mouse

# Constants
BESTEST_PATH = os.path.join("./nets", "bestest_mouse.pkl")
LATEST_PATH = os.path.join("./nets", "latest_mouse.pkl")
DASHBOARD_WIDTH = 350
MIN_SCREEN_HEIGHT = 550
FPS = 10
SPEED_MULTIPLIER = 3  # Steps per frame when speed mode is active


class SimulationMode:
    BEST = "best"
    USER_CONTROLLED = "user_controlled"
    TRAINING = "training"

def load_best_mouse():
    if not os.path.exists(BESTEST_PATH):
        raise FileNotFoundError(f"File not found: {BESTEST_PATH}")

    with open(BESTEST_PATH, "rb") as f:
        mouse = pickle.load(f)

    return mouse


def move_with_network(maze, mouse):
    """Move mouse based on neural network output."""
    inputs = mouse.get_inputs(maze)
    outputs = mouse.net.activate(inputs)
    action = outputs.index(max(outputs))
    mouse.act(action, maze)


def move_with_keys(event, maze, mouse):
    """Handle keyboard input for manual mouse control."""
    if event.type != pygame.KEYDOWN:
        return

    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
    }

    action = key_to_action.get(event.key)
    if action is not None:
        mouse.act(action, maze)


def get_window_caption(mode, mouse):
    if mode == SimulationMode.BEST:
        return "Micromouse Neuroevolution - Best Mouse"
    elif mode == SimulationMode.USER_CONTROLLED:
        return "Micromouse Neuroevolution - Manual Control"
    else:
        return f"Micromouse Neuroevolution - Generation {mouse.generation}"


def setup_screen(maze):
    """Initialize and return pygame screen with appropriate dimensions."""
    maze_pixel_size = maze.size * graphics.CELL_SIZE
    screen_height = max(maze_pixel_size, MIN_SCREEN_HEIGHT)
    screen_width = maze_pixel_size + DASHBOARD_WIDTH

    screen = pygame.display.set_mode((screen_width, screen_height))
    maze_offset_y = (screen_height - maze_pixel_size) // 2

    return screen, maze_offset_y


def run(mouse=None, maze=None, configuration=None):
    """Run the simulation with the specified mouse and maze."""
    loader = MazeLoader()

    if not pygame.get_init():
        pygame.init()

    if maze is None:
        maze = loader.get_random_maze()

    mode = SimulationMode.TRAINING

    if mouse is None or mouse.genome is None:
        try:
            mouse = load_best_mouse()
            mode = SimulationMode.BEST
        except FileNotFoundError:
            mouse = Mouse()
            mode = SimulationMode.USER_CONTROLLED

    mouse.reset()

    # Setup display
    pygame.display.set_caption(get_window_caption(mode, mouse))
    screen, maze_offset_y = setup_screen(maze)
    clock = pygame.time.Clock()

    if (mouse.net is None and
            mode != SimulationMode.USER_CONTROLLED and
            mouse.genome is not None):
        mouse.net = neat.nn.RecurrentNetwork.create(mouse.genome, configuration)

    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if mode == SimulationMode.USER_CONTROLLED:
                move_with_keys(event, maze, mouse)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_k]:
            if mouse.alive:
                mouse.alive = False
            elif mode == SimulationMode.BEST:
                running = False

        if mouse.alive:
            steps_per_frame = SPEED_MULTIPLIER if keys[pygame.K_s] else 1

            for _ in range(steps_per_frame):
                if mouse.alive and mode != SimulationMode.USER_CONTROLLED:
                    move_with_network(maze, mouse)
                else:
                    break
        elif mode != SimulationMode.BEST:
            running = False

        # Render
        screen.fill(graphics.BG_COLOR)

        maze_pixel_size = maze.size * graphics.CELL_SIZE
        graphics.draw_maze(screen, mouse, maze, offset_x=0, offset_y=maze_offset_y)
        graphics.draw_mouse(screen, mouse, offset_x=0, offset_y=maze_offset_y)
        graphics.draw_dashboard(
            screen=screen,
            x=maze_pixel_size,
            y=0,
            width=DASHBOARD_WIDTH,
            height=screen.get_height(),
            mouse=mouse,
            genome=mouse.genome,
            m=maze,
            best_simulation=(mode == SimulationMode.BEST)
        )

        pygame.display.flip()
        clock.tick(FPS)

    return mouse


def cleanup_pygame():
    if pygame.get_init():
        pygame.quit()


def main():
    """Main entry point for running the simulation."""
    try:
        local_dir = os.path.dirname(__file__)
        config_file = os.path.join(local_dir, 'config-neat.ini')

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )

        # Load and simulate best mouse
        mouse = load_best_mouse()
        run(mouse=mouse, configuration=config)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Running in manual control mode instead.")
        run()

    finally:
        cleanup_pygame()


if __name__ == '__main__':
    main()
