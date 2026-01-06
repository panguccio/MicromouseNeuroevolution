import maze as mz
from maze import Maze
from direction import Direction

# FITNESS WEIGHTS
COST_WEIGHT = 100 # 100
DISTANCE_WEIGHT = 150
NOVELTY_WEIGHT = 100
BONUS = 100

maze_size = 16

# MAX CONSTANTS
max_visits = 15
max_stuck_counter = 20
max_actions = 200

# COST CONSTANTS
STEP_COST = 0.01
TURN_COST = COST_WEIGHT / 200
COLLISION_COST = 1
STUCK_COST = COST_WEIGHT / 4


class Mouse:

    def __init__(self, start_position=(maze_size, 0), genome=None, gid=None, net=None,
                 generation="Cool", fitness=0):

        # STATUS AND CHARACTERISTICS
        self.alive = True
        self.start_position = start_position
        self.position = start_position
        self.direction = Direction.N
        self.arrived = False
        self.ahead_sight = 3
        self.lateral_sight = 3
        self.fate = "ALIVE"

        # MEMORY
        self.inner_maze = Maze(size=maze_size)
        self.visited_cells = set()
        self.visited_cells.add(start_position)
        self.inner_maze.add_visit(*start_position)
        self.saturated_cells = set()
        self.last_action = 0.5

        # FOR COST COMPUTATION
        self.actions = 0
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

        self.actions = 0
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
            self.sense_left(maze),
            self.sense_ahead(maze),
            self.sense_right(maze),
            self.get_direction(),
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
    # Other inputs
    # ---

    def visit_intensity(self):
        """how much the cell on the front of the mouse has been visited, max 15 times (ratio in [0, 1])"""
        row, column = self.position
        visits = self.inner_maze.get_visits(row + self.direction.dr, column + self.direction.dc)
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

    def get_direction(self):
        return self.direction.value / Direction.W.value

    def get_steps(self):
        return self.actions / max_actions

    def stuckness(self):
        if self.stuck_counter > max_visits:
            return 1
        return self.stuck_counter / max_visits


    # ---
    # Movement
    # ---

    def turn_left(self):
        self.turns += 1
        self.inner_maze.add_visit(*self.position)
        self.direction = self.direction.left

    def turn_right(self):
        self.turns += 1
        self.inner_maze.add_visit(*self.position)
        self.direction = self.direction.right

    def increment_path(self, position):
        if position not in self.visited_cells:
            self.visited_cells.add(position)
        self.inner_maze.add_visit(*position)

    def move_ahead(self, maze: Maze):
        r, c = self.position
        d = self.direction

        # checks if there's a wall
        if maze.has_wall(d, r, c):
            self.collisions += 1
            return

        # moves
        self.position = (r + self.direction.dr, c + self.direction.dc)
        self.increment_path(self.position)

    def act(self, action, maze: Maze):
        # executes an action
        self.last_action = action / 2
        self.actions += 1

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
            self.fate = "GOAL!"
            self.alive = False
            return

        # Controlla altre condizioni di terminazione
        if self.actions >= max_actions:
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
                fitness = BONUS + max(0, max_actions + 100 - self.actions - self.turns)
        else:
            distance_score = self.compute_distance_score(maze)
            performance = self.compute_performance()
            fitness = distance_score + performance
        return fitness

    def compute_distance_score(self, maze):
        max_distance = maze.minmax_distance_from_goal(self.start_position)
        closest_distance = min(maze.minmax_distance_from_goal(position) for position in self.visited_cells)
        distance_score = max_distance - closest_distance
        return int(distance_score * 10)

    def compute_performance(self):
        performance = 9
        if self.stuck:
            performance -= 3
        performance -= self.visit_rate_cost()
        performance -= self.turn_cost()
        performance -= self.collision_cost()
        return max(0, int(performance))

    def visit_rate_cost(self):
        visit_rate = self.actions / len(self.visited_cells)
        return (visit_rate > 2) + (visit_rate > 4) + (visit_rate > 8) + (visit_rate > 12) + (visit_rate > 14)

    def turn_cost(self):
        turn_rate = 100 * self.turns / max_actions
        return (turn_rate > 10) + (turn_rate > 20) + (turn_rate > 30) + (turn_rate > 50) + (turn_rate > 70)

    def collision_cost(self):
        return (self.collisions > 1) + (self.collisions > 3) + (self.collisions > 5)








