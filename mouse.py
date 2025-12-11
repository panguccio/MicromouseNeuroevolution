import maze as mz
from maze import Maze
from direction import Direction

# FITNESS WEIGHTS
COST_WEIGHT = .7
DISTANCE_WEIGHT = 2
NOVELTY_WEIGHT = .5
maze_size = 16


class Mouse:

    def __init__(self, start_position=(maze_size, 0), max_steps=maze_size ** 2, genome=None, gid=None, net=None,
                 generation=0, fitness=0):
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
        self.visited = set()
        self.visited.add(start_position)

        # FOR COST COMPUTATION
        self.steps = 0
        self.max_steps = max_steps
        self.cost = 0
        # COST CONSTANTS
        # self.STEP_COST = 0.01
        # self.TURN_COST = 0
        # self.COLLISION_COST = 1
        # self.STUCK_COST = 20
        # FOR REVISITED CHECK

        # STUCK CONTROL
        self.last_action = None
        self.last_inputs = []
        self.last_position = self.position
        self.stuck_counter = 0
        self.stuck = False
        self.max_stuck_counter = 20

        # GENETICS
        self.genome = genome
        self.gid = gid
        self.net = net
        self.distance_scores_values = []
        self.novelty_scores_values = []
        self.costs = []
        self.generation = generation

        # FITNESS
        self.fitness_values = []
        self.fitness = fitness
        # self.MAX_NOVELTY_SCORE = 14

    def reset(self):
        self.alive = True
        self.position = self.start_position
        self.direction = Direction.N
        self.arrived = False
        self.steps = 0
        self.cost = 0
        self.visited = set()
        self.visited.add(self.start_position)
        self.last_action = None
        self.last_inputs = []
        self.last_position = self.position
        self.stuck_counter = 0
        self.stuck = False

    # ---
    # Input processing
    # ---

    def get_inputs(self, maze: Maze):
        # values are normalized
        return [
            self.proximity(maze),
            self.sense_left(maze),
            self.sense_ahead(maze),
            self.sense_right(maze)
        ]

    # ---
    # Sensor logic
    # ---

    def proximity(self, maze: Maze):
        return (maze.size // 2 - 1 - maze.minmax_distance_from_goal(self.position)) / (maze.size // 2 - 1)

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
        if distance == 1:
            self.inner_maze.add_wall(direction, *self.position)
        return 1 - distance / sight

    # ---
    # Movement
    # ---

    def turn_left(self):
        # self.cost += self.TURN_COST
        self.direction = self.direction.left

    def turn_right(self):
        # self.cost += self.TURN_COST
        self.direction = self.direction.right

    def increment_path(self, position):
        self.steps += 1
        # self.cost += self.STEP_COST
        if position not in self.visited:
            self.visited.add(position)

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
        if self.steps >= self.max_steps:
            self.alive = False
            return

        self.check_stuck()

        if self.stuck_counter > self.max_stuck_counter:
            self.stuck = True
            self.alive = False
            # self.cost += self.STUCK_COST
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

    def check_loop(self):
        # to penalize useless cycles
        revisit_ratio = self.steps / len(self.visited)  # n of times a cell is revisited on average
        if revisit_ratio >= self.max_stuck_counter / 5:
            self.stuck = True
            min_value = self.max_stuck_counter / 5
            max_value = self.max_steps / 2  # if its
            normalized_ratio = (revisit_ratio - min_value) / (max_value - min_value)
            # so that the cost is between 10 (better than stuck) and 20 (like stuck)
            # self.cost += self.STUCK_COST / 2 * (1 + normalized_ratio)

    # ---
    # Fitness
    # ---

    def compute_distance_score(self, distance):
        if self.arrived:
            score = 100 + (self.max_steps - self.steps)
        else:
            # should be between 1 and 16
            score = maze_size - distance
        self.distance_scores_values.append(score)

    def compute_cost(self):
        self.check_loop()
        self.costs.append(0.5 if self.stuck else 2)

    def compute_novelty_score(self, others_positions, k):
        # should be between 0 and 32 (but more likely near 0 with a max of 16)
        distances = [mz.man_distance(self.position, position) for position in others_positions]
        k_nearest = sorted(distances)[1:min(k + 1, len(distances))]
        score = sum(k_nearest) / len(k_nearest)
        self.novelty_scores_values.append(score)

    def compute_fitness_score(self, distance_weight=DISTANCE_WEIGHT, novelty_weight=NOVELTY_WEIGHT,
                              cost_weight=COST_WEIGHT):
        for i in range(len(self.distance_scores_values)):
            distance_score = self.distance_scores_values[i]
            cost = self.costs[i]
            novelty_score = self.novelty_scores_values[i]

            fitness = (distance_weight * distance_score + novelty_weight * novelty_score) * cost_weight * cost
            self.fitness_values.append(max(fitness, 0))
        self.fitness = sum(self.fitness_values) / len(self.fitness_values)
