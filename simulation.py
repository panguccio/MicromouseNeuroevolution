import pygame
from maze_loader import MazeLoader
from maze import Maze
from direction import Direction
from mouse import Mouse  # oppure incolla la classe dentro lo script

CELL_SIZE = 30
BG_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)
MOUSE_COLOR = (255, 255, 255)
WALL_THICKNESS = 2
MARGIN = 2
MOUSE_IMG = pygame.image.load("mouse.png")
MOUSE_IMG = pygame.transform.scale(MOUSE_IMG, (CELL_SIZE / 1.5, CELL_SIZE / 1.5))
pygame.transform.rotate(MOUSE_IMG, 180)


loader = MazeLoader()


def draw_maze(screen, maze: Maze):
    size = maze.size

    for r in range(size):
        for c in range(size):
            x = c * CELL_SIZE
            y = r * CELL_SIZE

            # Parete Nord
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

    angle = mouse.direction.angle

    rotated = pygame.transform.rotate(MOUSE_IMG, angle)
    rect = rotated.get_rect(center=(x, y))
    screen.blit(rotated, rect)


def try_move(mouse: Mouse, maze: Maze):
    r, c = mouse.position
    d = mouse.direction

    # Se c'è un muro, blocca
    if maze.has_wall(d, r, c):
        return

    # Move allowed
    mouse.move_ahead()

def main():
    pygame.init()
    maze = loader.get_random_maze()
    maze.print_grid()
    mouse = Mouse()

    screen = pygame.display.set_mode(
        (maze.size * CELL_SIZE + MARGIN, maze.size * CELL_SIZE + MARGIN)
    )
    pygame.display.set_caption("Micromouse Maze + Mouse")

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Movimento con frecce
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    try_move(mouse, maze)

                if event.key == pygame.K_LEFT:
                    mouse.turn_left()

                if event.key == pygame.K_RIGHT:
                    mouse.turn_right()


        draw_maze(screen, maze)
        draw_mouse(screen, mouse)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()



