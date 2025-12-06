import pygame
from maze_loader import MazeLoader
from maze import Direction, Maze
from mouse import Mouse  # oppure incolla la classe dentro lo script

CELL_SIZE = 30
BG_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)
MOUSE_COLOR = (255, 255, 255)
WALL_THICKNESS = 2
MARGIN = 2

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
    radius = CELL_SIZE // 4

    pygame.draw.circle(screen, MOUSE_COLOR, (x, y), radius)


def try_move(mouse: Mouse, maze: Maze):
    r, c = mouse.position
    d = mouse.direction

    # Se c'è un muro, blocca
    if maze.has_wall(d, r, c):
        return

    # Move allowed
    mouse.position[0] += d.dr
    mouse.position[1] += d.dc


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
                    mouse.redirect(Direction.N)
                    try_move(mouse, maze)

                if event.key == pygame.K_DOWN:
                    mouse.redirect(Direction.S)
                    try_move(mouse, maze)

                if event.key == pygame.K_LEFT:
                    mouse.redirect(Direction.W)
                    try_move(mouse, maze)

                if event.key == pygame.K_RIGHT:
                    mouse.redirect(Direction.E)
                    try_move(mouse, maze)

        draw_maze(screen, maze)
        draw_mouse(screen, mouse)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()



