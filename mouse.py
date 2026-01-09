import neat
from maze import Maze
from direction import Direction

BONUS = 500
maze_size = 16

# MAX CONSTANTS
max_visits = 15
max_steps = maze_size ** 2
max_stuck_counter = max_steps // 3


class Mouse:

    def __init__(self, start_position=(maze_size, 0), genome=None, gid=None, net=None,
                 generation="X", fitness=0):

        # STATUS AND CHARACTERISTICS
        self.alive = True
        self.start_position = start_position
        self.position = start_position
        self.direction = Direction.N
        self.arrived = False
        self.sight = 1
        self.fate = "ALIVE"

        # MEMORY
        self.path_sequence = []
        self.path_sequence.append(self.start_position)
        self.saturated_cells = set()

        # FOR COST COMPUTATION
        self.steps = 0
        self.collisions = 0
        self.bumped = False

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

        self.path_sequence = []
        self.path_sequence.append(self.start_position)

        self.direction = Direction.N
        self.arrived = False

        self.steps = 0
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
            self.has_bumped(),
            self.relative_position_x(maze),
            self.relative_position_y(maze),
            self.proximity(maze),
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

    def has_bumped(self):
        return 1 if self.bumped else 0

    def proximity(self, maze: Maze):
        """in which 'circle' it's positioned the mouse, that is, how close it's to the center"""
        return (maze_size // 2 - 1 - maze.range_distance_from_goal(self.position)) / (maze_size // 2 - 1)

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
        self.path_sequence.append(position)

    def move(self, d: Direction, maze: Maze):
        self.direction = d
        r, c = self.position

        # checks if there's a wall
        if maze.has_wall(d, r, c):
            self.collisions += 1
            self.bumped = True
            return

        # moves
        self.position = (r + self.direction.dr, c + self.direction.dc)
        self.increment_path(self.position)

    def act(self, direction, maze: Maze):
        if self.bumped:
            self.bumped = False
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

    def visited_cells(self):
        return set(self.path_sequence)

    # ---
    # Maze exploration
    # ---

    def explore(self, config, maze):
        self.reset()
        self.net = neat.nn.RecurrentNetwork.create(self.genome, config)
        while self.alive:
            inputs = self.get_inputs(maze)
            outputs = self.net.activate(inputs)
            action = outputs.index(max(outputs))
            self.act(action, maze)
        self.compute_maze_score(maze)

    def stats(self):
        import maze
        genetics = f"\tgeneration: {self.generation}; gid: {self.gid}\n"
        position = f"\tlast position: {self.position} -> {maze.manhattan_distance_from_goal(self.position)} from goal\n"
        fitness = f"\tfitness: {self.fitness} = {self.fitness_values}\n"
        status = f"\tarrived? {self.arrived}. stuck? {self.stuck}. \n"
        path = f"\tsteps: {self.steps}, num visited: {len(self.visited_cells())}\n"
        costs = f"visits/cell: {self.visit_rate()}; coverage: {(len(self.visited_cells())/(maze_size**2)):.2f}%; collisions: {self.collisions}\n"
        return genetics + status + position + fitness + path + costs

    # ---
    # Fitness
    # ---

    def compute_fitness_score(self):
        self.fitness = sum(self.fitness_values) / len(self.fitness_values)
        self.genome.fitness = self.fitness
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
            fitness = len(self.visited_cells()) / (1 + self.collisions + self.visit_rate())
        return fitness

    def visit_rate(self):
        visit_rate = (self.steps / len(self.visited_cells()))
        return visit_rate