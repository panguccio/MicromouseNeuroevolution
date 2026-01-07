import maze as mz
from maze import Maze
from direction import Direction

BONUS = 150

maze_size = 16

# MAX CONSTANTS
max_visits = 15
max_steps = maze_size ** 2
max_stuck_counter = max_steps // 3

class Mouse:

    def __init__(self, start_position=(maze_size, 0), genome=None, gid=None, net=None,
                 generation="Cool", fitness=0):

        # STATUS AND CHARACTERISTICS
        self.alive = True
        self.start_position = start_position
        self.position = start_position
        self.direction = Direction.N
        self.arrived = False
        self.sight = 3
        self.fate = "ALIVE"

        # MEMORY
        self.inner_maze = Maze(size=maze_size)
        self.visited_cells = set()
        self.visited_cells.add(start_position)
        self.inner_maze.add_visit(*start_position)
        self.saturated_cells = set()
        self.last_action = 1/4

        # FOR COST COMPUTATION
        self.steps = 0
        self.turns = 0
        self.collisions = 0

        # STUCK CONTROL
        self.last_position = self.position
        self.stuck_counter = 0
        self.stuck = False

        # GENETICS
        self.genome = genome
        self.gid = gid
        self.net = net
        self.generation = generation

        # FITNESS
        self.fitness_values = []
        self.fitness = fitness

    def reset(self):
        self.alive = True
        self.position = self.start_position
        self.fate = "ALIVE"

        self.inner_maze = Maze(size=maze_size)
        self.visited_cells = set()
        self.visited_cells.add(self.start_position)
        self.last_action = 0.5

        self.direction = Direction.N
        self.arrived = False

        self.steps = 0
        self.turns = 0
        self.collisions = 0

        self.last_position = self.position
        self.stuck_counter = 0
        self.stuck = False

        if self.net is not None: self.net.reset()

    # ---
    # Input processing
    # ---

    def get_inputs(self, maze: Maze):
        # values are normalized
        inputs = [
            self.sense_north(maze),
            self.sense_east(maze),
            self.sense_south(maze),
            self.sense_west(maze),
            self.relative_position_x(maze),
            self.relative_position_y(maze),
            self.proximity(maze),
            self.get_steps(),
            self.stuckness(),
            self.visit_intensity(),
            self.last_action
        ]
        return inputs

    # ---
    # Sensor logic
    # ---

    def sense_north(self, maze: Maze):
        return self.sense(maze, Direction.N, self.sight)

    def sense_east(self, maze: Maze):
        return self.sense(maze, Direction.E, self.sight)

    def sense_south(self, maze: Maze):
        return self.sense(maze, Direction.S, self.sight)

    def sense_west(self, maze: Maze):
        return self.sense(maze, Direction.W, self.sight)

    def sense(self, maze: Maze, direction, sight):
        distance = maze.first_wall(direction, *self.position, sight)
        if distance is None:
            return 0
        return 1 - distance / sight

    # ---
    # Other inputs
    # ---

    def visit_intensity(self):
        """how much the cell the mouse is on has been visited, max 15 times (ratio in [0, 1])"""
        row, column = self.position
        visits = self.inner_maze.get_visits(row, column)
        return float(visits) / max_visits

    def proximity(self, maze: Maze):
        """in which 'circle' it's positioned the mouse, that is, how close it's to the center"""
        return (maze_size // 2 - 1 - maze.minmax_distance_from_goal(self.position)) / (maze_size // 2 - 1)

    def relative_position_x(self, maze):
        max_distance = maze.x_distance_from_goal(self.start_position)
        return (max_distance - maze.x_distance_from_goal(self.position)) / max_distance

    def relative_position_y(self, maze):
        max_distance = maze.y_distance_from_goal(self.start_position)
        return (max_distance - maze.y_distance_from_goal(self.position)) / max_distance

    def get_steps(self):
        return self.steps / max_steps

    def stuckness(self):
        if self.stuck_counter > max_visits:
            return 1
        return self.stuck_counter / max_visits


    # ---
    # Movement
    # ---

    def increment_path(self, position):
        if position not in self.visited_cells:
            self.visited_cells.add(position)
        self.inner_maze.add_visit(*position)

    def move(self, d: Direction, maze: Maze):
        self.direction = d
        r, c = self.position

        # checks if there's a wall
        if maze.has_wall(d, r, c):
            self.collisions += 1
            return

        # moves
        self.position = (r + self.direction.dr, c + self.direction.dc)
        self.increment_path(self.position)

    def act(self, direction, maze: Maze):

        # executes an action
        self.last_action = (direction + 1) / 4
        self.steps += 1

        self.move(Direction(direction), maze)

        # Controlla se è arrivato al goal
        if maze.is_in_goal(self.position):
            self.arrived = True
            self.fate = "GOAL!"
            self.alive = False
            return

        # Controlla altre condizioni di terminazione
        if self.steps >= max_steps:
            self.fate = "TIMEOUT"
            self.alive = False
            return

        self.check_stuck()

        if self.stuck_counter > max_stuck_counter:
            self.stuck = True
            self.fate = "STUCK"
            self.alive = False
            # self.cost += STUCK_COST
            return

        self.last_position = self.position

    # ---
    # Checks
    # ---

    def check_stuck(self):
        # Controlla se è bloccato
        if self.position == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def check_fate(self):
        if self.fate == "ALIVE" and not self.alive: return "CRASHED"
        return self.fate

    # ---
    # Fitness
    # ---

    def compute_fitness_score(self):
        self.fitness = sum(self.fitness_values) / len(self.fitness_values)
        return self.fitness

    def compute_maze_score(self, maze):
        fitness = self.compute_fitness(maze)
        self.fitness_values.append(max(fitness, 0))

    def update_maze_score(self, maze, i):
        fitness = self.compute_fitness(maze)
        if len(self.fitness_values) <= i:
            self.fitness_values.append(max(fitness, 0))
        else:
            self.fitness_values[i] = max(fitness, 0)

    def compute_fitness(self, maze):
        if self.arrived:
                fitness = BONUS + max(0, max_steps - self.steps)
        else:
            distance_score = self.compute_distance_score(maze)
            performance = self.compute_performance()
            fitness = distance_score + performance
        return fitness

    def compute_distance_score(self, maze):
        max_distance = maze.man_distance_from_goal(self.start_position)
        closest_distance = min(maze.man_distance_from_goal(position) for position in self.visited_cells)
        distance_score = max_distance - closest_distance
        return int(distance_score * 10)

    def compute_performance(self):
        performance = 9
        if self.stuck:
            performance -= 2
        performance -= self.visit_rate_cost()
        performance -= self.collision_cost()
        return max(0, int(performance))

    def visit_rate_cost(self):
        visit_rate = self.steps / len(self.visited_cells)
        return (visit_rate > 4) + (visit_rate > 8) + (visit_rate > 12) + (visit_rate > 15)

    def collision_cost(self):
        return (self.collisions > 10) + (self.collisions > 50) + (self.collisions > 120) + (self.collisions > 200)








