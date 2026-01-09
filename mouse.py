import math

import neat

import math

import neat

import maze as mz
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
        self.path_sequence = []
        self.path_sequence.append(self.start_position)
        self.inner_maze.add_visit(*start_position)
        self.saturated_cells = set()

        # FOR COST COMPUTATION
        self.steps = 0
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
        self.path_sequence = []
        self.path_sequence.append(self.start_position)
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
            self.visit_intensity()
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

    def phenotype_distance(self, mouse):
        """
        returns the Levenshtein distance between the sequence of cells the 2 mice explored.
        """
        def levenshtein(s1, s2):
            def head(s):
                return s[0]
            def tail(s):
                return s[1:]

            if len(s1) == 0:
                return len(s2)
            if len(s2) == 0:
                return len(s1)
            if head(s1) == head(s2):
                return levenshtein(tail(s1), tail(s2))
            else:
                return 1 + min(levenshtein(tail(s1), s2), levenshtein(s1, tail(s2)), levenshtein(tail(s1), tail(s2)))

        return levenshtein(self.path_sequence, mouse.path_sequence)


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
        genetics = f"\tgeneration: {self.generation}; gid: {self.gid}\n"
        position = f"\tlast position: {self.position} -> {self.inner_maze.manhattan_distance_from_goal(self.position)} from goal\n"
        fitness = f"\tfitness: {self.fitness} = {self.fitness_values}\n"
        status = f"\tarrived? {self.arrived}. stuck? {self.stuck}. \n"
        path = f"\tsteps: {self.actions}, num visited: {len(self.visited_cells())}\n"
        costs = f"visited cost: {self.actions / len(self.visited_cells())} -> {self.visit_rate_cost()}, collisions: {self.collisions} -> {self.collision_cost()}\n"
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
            # fitness = fitness * (1 + self.behaviour_delta())
        else:
            fitness = self.compute_distance_score(maze)
            # fitness = fitness * (1 + self.behaviour_delta())
        return fitness

    def compute_distance_score(self, maze):
        max_distance = maze.manhattan_distance_from_goal(self.start_position)
        closest_distance = min(maze.manhattan_distance_from_goal(position) for position in self.visited_cells())
        score_closest_distance = (max_distance - closest_distance) # / self.coverage()
        return 10 * score_closest_distance + int(10 * (1 - len(self.path_sequence)/max_steps))

    def behaviour_delta(self):
        """
        value in [-0.1, 0.1] to modify the fitness of ±10% for the mouse behaviour.
        """
        score = 1
        if self.stuck:
            score -= 0.25
        score -= 0.25 * self.turn_cost()
        score -= 0.25 * self.visit_rate_cost()
        score -= 0.25 * self.collision_cost()
        return score / 5 - 0.1

    def turn_cost(self):
        turn_rate = self.turns / (1 + self.actions)
        return 1 / (1 + math.exp(-8 * (turn_rate - 0.55)))

    def visit_rate_cost(self):
        visit_rate = self.actions / len(self.visited_cells())
        return visit_rate / (visit_rate + 3)

    def collision_cost(self):
        return self.collisions / (self.collisions + 2)

    def coverage(self):
        coverage_rate = len(self.visited_cells()) / (maze_size ** 2)
        return 1 + (1 - coverage_rate) ** 2.8

