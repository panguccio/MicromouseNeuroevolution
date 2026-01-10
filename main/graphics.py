import os

import pygame

from main import mouse as mice, maze
from main.direction import Direction
from main.maze import Maze
from main.mouse import Mouse

CELL_SIZE = 30
BG_COLOR = (20, 20, 20)
MAZE_BG_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)
WALL_THICKNESS = 2
UI_BG_COLOR = (40, 40, 40)
TEXT_COLOR = (240, 240, 240)
ACCENT_COLOR = (255, 50, 50)
SUCCESS_COLOR = (0, 255, 0)
POS_VAL_COLOR = (0, 255, 100)
NEG_VAL_COLOR = (255, 80, 80)
MOUSE_IMG_PATH = "./images/mouse.png"
input_labels = ["N", "E", "S", "W", "X", "Y", "P"]
num_inputs = len(input_labels)
num_outputs = 4


if os.path.exists(MOUSE_IMG_PATH):
    MOUSE_IMG = pygame.image.load(MOUSE_IMG_PATH)
    MOUSE_IMG = pygame.transform.scale(MOUSE_IMG, (int(CELL_SIZE * 0.7), int(CELL_SIZE * 0.7)))
else:
    MOUSE_IMG = pygame.Surface((int(CELL_SIZE * 0.7), int(CELL_SIZE * 0.7)))
    MOUSE_IMG.fill((255, 255, 255))


def draw_maze(screen, mouse, m: Maze, offset_x=0, offset_y=0):
    if mouse.alive:
        wall_color = WALL_COLOR
    else:
        wall_color = TEXT_COLOR
    size = m.size
    pygame.draw.rect(screen, MAZE_BG_COLOR, (offset_x, offset_y, size * CELL_SIZE, size * CELL_SIZE))

    for r in range(size):
        for c in range(size):
            x = c * CELL_SIZE + offset_x
            y = r * CELL_SIZE + offset_y
            if m.has_wall(Direction.N, r, c):
                pygame.draw.line(screen, wall_color, (x, y), (x + CELL_SIZE, y), WALL_THICKNESS)
            if m.has_wall(Direction.S, r, c):
                pygame.draw.line(screen, wall_color, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), WALL_THICKNESS)
            if m.has_wall(Direction.W, r, c):
                pygame.draw.line(screen, wall_color, (x, y), (x, y + CELL_SIZE), WALL_THICKNESS)
            if m.has_wall(Direction.E, r, c):
                pygame.draw.line(screen, wall_color, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), WALL_THICKNESS)

def draw_mouse(screen, mouse: Mouse, offset_x=0, offset_y=0):
    r, c = mouse.position
    x = c * CELL_SIZE + CELL_SIZE // 2 + offset_x
    y = r * CELL_SIZE + CELL_SIZE // 2 + offset_y

    if isinstance(MOUSE_IMG, pygame.Surface):
        rotated = pygame.transform.rotate(MOUSE_IMG, mouse.direction.angle)
        rect = rotated.get_rect(center=(x, y))
        screen.blit(rotated, rect)
    else:
        pygame.draw.circle(screen, (255, 255, 255), (x, y), CELL_SIZE // 3)

def draw_text(screen, text, x, y, size=18, color=TEXT_COLOR, bold=False):
    font = pygame.font.SysFont('Arial', size, bold=bold)
    surface = font.render(str(text), True, color)
    screen.blit(surface, (x, y))
    return surface.get_height()

def get_death_reason(mouse):
    if mouse.alive: return "ALIVE"
    if mouse.steps >= mice.MAX_STEPS: return "TIMEOUT"
    if mouse.arrived: return "GOAL!"
    return "CRASHED"


def draw_dashboard(screen, x, y, width, height, mouse, genome, m, best_simulation):
    pygame.draw.rect(screen, UI_BG_COLOR, (x, y, width, height))
    pygame.draw.line(screen, ACCENT_COLOR, (x, y), (x, height), 2)

    padding = 20
    current_y = y + padding

    draw_text(screen, "MICROMOUSE AI", x + padding, current_y, 26, ACCENT_COLOR, bold=True)
    current_y += 35
    draw_text(screen, f"Map: {m.name}", x + padding, current_y, 16, (150, 150, 150))
    current_y += 30

    reason_color = ACCENT_COLOR if not mouse.alive else TEXT_COLOR
    if mouse.genome is not None:
        fitness = mouse.genome.fitness
    else:
        fitness = 0
    stats = [
        ("Genome", mouse.gid),
        ("Status", get_death_reason(mouse), SUCCESS_COLOR if mouse.arrived else reason_color),
        ("Distance", maze.manhattan_distance_from_goal(mouse.position)),
        ("Current fitness", f"{fitness:.2f}"),
        ("Steps", f"{mouse.steps}"),
        ("Visits per cell", f"{mouse.steps / len(mouse.visited_cells):.2f}"),
        ("Collisions", f"{mouse.collisions}"),
    ]

    for item in stats:
        label, val = item[0], item[1]
        col = item[2] if len(item) > 2 else TEXT_COLOR
        draw_text(screen, f"{label}:", x + padding, current_y, 16, (180, 180, 180))
        draw_text(screen, str(val), x + width - padding - 80, current_y, 16, col, bold=True)
        current_y += 20

    current_y += 10
    pygame.draw.line(screen, (80, 80, 80), (x + 10, current_y), (x + width - 10, current_y), 1)
    current_y += 20

    draw_text(screen, "Inputs", x + padding, current_y, 14, ACCENT_COLOR)
    current_y += 50

    inputs = mouse.get_inputs(m)
    bar_width = (width - 2 * padding) / num_inputs

    for i, val in enumerate(inputs[:num_inputs]):
        bar_x = x + padding + i * bar_width

        draw_text(screen, input_labels[i], bar_x + bar_width // 2 - 5, current_y - 20, 14, (200, 200, 200),
                  bold=True)

        bar_h = 15
        intensity = min(max(val, 0), 1)
        col_r = int(255 * intensity)
        col_g = int(255 * (1 - intensity))

        pygame.draw.rect(screen, (30, 30, 30), (bar_x + 5, current_y, bar_width - 10, bar_h))
        pygame.draw.rect(screen, (col_r, col_g, 0), (bar_x + 5, current_y, (bar_width - 10) * intensity, bar_h))
        pygame.draw.rect(screen, (100, 100, 100), (bar_x + 5, current_y, bar_width - 10, bar_h), 1)
    current_y += 30

    current_y += 10

    draw_text(screen, "Neural Network", x + padding, current_y, 14, ACCENT_COLOR)
    current_y += 20

    net_height = height - current_y - 40
    draw_network_dynamic(screen, genome, x + 10, current_y, width - 20, net_height)

    info_y = height - 25
    if not mouse.alive and best_simulation:
        draw_text(screen, "PRESS [K] TO CLOSE", x + padding, info_y, 14, ACCENT_COLOR, bold=True)
    elif not mouse.alive and not best_simulation:
        draw_text(screen, "Loading next generation...", x + padding, info_y, 14, ACCENT_COLOR, bold=True)
    else:
        draw_text(screen, "[S] Hold to Speed Up   [K] Kill", x + padding, info_y, 14, (150, 150, 150))


def draw_network_dynamic(screen, genome, x, y, w, h):
    if genome is None: return

    all_nodes = list(genome.nodes.keys())
    input_nodes = [-i for i in range(1, num_inputs + 1)]
    output_nodes = [i for i in range(0, num_outputs)]
    hidden_nodes = [n for n in all_nodes if n not in input_nodes and n not in output_nodes]

    node_pos = {}
    radius = 7

    for i, nid in enumerate(input_nodes):
        nx = x + 30
        ny = y + (h / (len(input_nodes) + 1)) * (i + 1)
        node_pos[nid] = (nx, ny)

    for i, nid in enumerate(output_nodes):
        nx = x + w - 30
        ny = y + (h / (len(output_nodes) + 1)) * (i + 1)
        node_pos[nid] = (nx, ny)

    if hidden_nodes:
        max_per_column = 10
        num_columns = (len(hidden_nodes) - 1) // max_per_column + 1

        for i, nid in enumerate(hidden_nodes):
            col = i // max_per_column
            row = i % max_per_column
            layer_x = x + (w / (num_columns + 1)) * (col + 1)
            ny = y + (h / (min(len(hidden_nodes), max_per_column) + 1)) * (row + 1)
            node_pos[nid] = (layer_x, ny)

    for (in_node, out_node), conn in genome.connections.items():
        if conn.enabled and in_node in node_pos and out_node in node_pos:
            start = node_pos[in_node]
            end = node_pos[out_node]
            is_recurrent = False
            if in_node in hidden_nodes and out_node in hidden_nodes:
                is_recurrent = True
            elif out_node in input_nodes or (out_node in hidden_nodes and in_node in output_nodes):
                is_recurrent = True

            color = POS_VAL_COLOR if conn.weight > 0 else NEG_VAL_COLOR
            width_line = max(1, min(5, int(abs(conn.weight) * 3)))

            if is_recurrent:
                draw_dashed_line(screen, color, start, end, width_line)
            else:
                pygame.draw.line(screen, color, start, end, width_line)

            if in_node == out_node:
                nx, ny = node_pos[in_node]
                pygame.draw.circle(screen, color, (int(nx + radius + 5), int(ny)), radius // 2, width_line)

    # Disegna NODI
    for nid, (nx, ny) in node_pos.items():
        color = (150, 150, 150)  # Hidden
        if nid in input_nodes:
            color = (50, 100, 255)  # Input
        elif nid in output_nodes:
            color = (255, 50, 50)  # Output

        pygame.draw.circle(screen, color, (int(nx), int(ny)), radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(nx), int(ny)), radius, 1)


def draw_dashed_line(screen, color, start, end, width, dash_length=5):
    """Disegna una linea tratteggiata per evidenziare connessioni ricorrenti"""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx ** 2 + dy ** 2) ** 0.5

    if distance == 0:
        return

    dashes = int(distance / dash_length)
    for i in range(0, dashes, 2):
        start_pos = (
            x1 + (dx * i / dashes),
            y1 + (dy * i / dashes)
        )
        end_pos = (
            x1 + (dx * (i + 1) / dashes),
            y1 + (dy * (i + 1) / dashes)
        )
        pygame.draw.line(screen, color, start_pos, end_pos, width)

