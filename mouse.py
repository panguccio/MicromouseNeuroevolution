import maze as mz
from maze import Maze
from direction import Direction

# FITNESS WEIGHTS
COST_WEIGHT = 100 # 100
DISTANCE_WEIGHT = 150
NOVELTY_WEIGHT = 100
BONUS = 250

maze_size = 16

# MAX CONSTANTS
max_visits = 15
max_stuck_counter = 300
max_steps = 2 * maze_size ** 2

# COST CONSTANTS
STEP_COST = 0.01
TURN_COST = COST_WEIGHT / 200
COLLISION_COST = 1
STUCK_COST = COST_WEIGHT / 4


class Mouse:

    def __init__(self, start_position=(maze_size, 0), genome=None, gid=None, net=None,
                 generation=0, fitness=0):
        global max_steps
        self.max_steps = max_steps

        # STATUS AND CHARACTERISTICS
        self.alive = True
        self.start_position = start_position
        self.position = start_position
        self.direction = Direction.N
        self.arrived = False
        self.ahead_sight = 3
        self.lateral_sight = 3

        # MEMORY
        self.inner_maze = Maze(size=maze_size)
        self.visited_cells = set()
        self.visited_cells.add(start_position)
        self.inner_maze.add_visit(*start_position)
        self.saturated_cells = set()

        # FOR COST COMPUTATION
        self.steps = 0
        self.cost = 0

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
        self.direction = Direction.N
        self.arrived = False
        self.steps = 0
        self.cost = 0
        self.last_position = self.position
        self.stuck_counter = 0
        self.stuck = False
        self.inner_maze = Maze(size=maze_size)
        if self.net is not None: self.net.reset()

    # ---
    # Input processing
    # ---

    def get_inputs(self, maze: Maze):
        # values are normalized
        return [
            self.proximity(maze),
            self.sense_left(maze),
            self.sense_ahead(maze),
            self.sense_right(maze),
            self.visit_intensity()
        ]

    # ---
    # Sensor logic
    # ---

    def visit_intensity(self):
        """how much the cell on the front of the mouse has been visited, max 15 times (ratio in [0, 1])"""
        row, column = self.position
        visits = self.inner_maze.get_visits(row + self.direction.dr, column + self.direction.dc)
        return visits / max_visits

    def proximity(self, maze: Maze):
        """in which 'circle' it's positioned the mouse, that is, how close it's to the center"""
        return (maze_size // 2 - 1 - maze.minmax_distance_from_goal(self.position)) / (maze_size // 2 - 1)

    def sense_ahead(self, maze: Maze):
        return self.sense(maze, self.direction, self.ahead_sight)

    def sense_left(self, maze: Maze):
        return self.sense(maze, self.direction.left, self.lateral_sight)

    def sense_right(self, maze: Maze):
        return self.sense(maze, self.direction.right, self.lateral_sight)

    def sense(self, maze: Maze, direction, sight):
        if not self.alive:
            return 0
        distance = maze.first_wall(direction, *self.position, sight)
        if distance is None:
            return 0
        return 1 - distance / sight

    # ---
    # Movement
    # ---

    def turn_left(self):
        self.cost += TURN_COST
        self.inner_maze.add_visit(*self.position)
        self.direction = self.direction.left

    def turn_right(self):
        self.cost += TURN_COST
        self.inner_maze.add_visit(*self.position)
        self.direction = self.direction.right

    def increment_path(self, position):
        self.steps += 1
        # self.cost += self.STEP_COST
        if position not in self.visited_cells:
            self.visited_cells.add(position)
        self.inner_maze.add_visit(*position)

    def move_ahead(self, maze: Maze):
        r, c = self.position
        d = self.direction

        # checks if there's a wall
        if maze.has_wall(d, r, c):
            # self.cost += self.COLLISION_COST
            return

        # moves
        self.position = (r + self.direction.dr, c + self.direction.dc)
        self.increment_path(self.position)

    def act(self, action, maze: Maze):
        # executes an action
        match action:
            case 0:
                self.turn_left()
            case 1:
                self.move_ahead(maze)
            case 2:
                self.turn_right()

        # Controlla se è arrivato al goal
        if maze.is_in_goal(self.position):
            self.arrived = True
            self.alive = False
            return

        # Controlla altre condizioni di terminazione
        if self.steps >= max_steps:
            self.alive = False
            return

        self.check_stuck()

        if self.stuck_counter > max_stuck_counter:
            self.stuck = True
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

    # ---
    # Fitness
    # ---

    def compute_fitness_score(self):
        if self.arrived:
            self.fitness = BONUS - self.steps
        else:
            self.fitness = sum(self.fitness_values) / len(self.fitness_values)
        return self.fitness

    def compute_maze_score(self, maze):
        distance_score = self.compute_distance_score(maze)
        performance = self.compute_performance()
        fitness = distance_score + performance
        self.fitness_values.append(max(fitness, 0))

    def compute_distance_score(self, maze):
        max_distance = maze.minmax_distance_from_goal(self.start_position)
        distance_score = max_distance - maze.minmax_distance_from_goal(self.position)
        return distance_score * 10

    def compute_performance(self):
        performance = 9
        if self.stuck:
            performance -= 2
        performance -= self.visit_rate() // 2
        return performance

    def visit_rate(self):
        return max(self.steps / len(self.visited_cells), max_visits) # in 1 - 15








