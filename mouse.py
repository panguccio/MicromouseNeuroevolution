from maze import Maze
from direction import Direction


class Mouse:

    def __init__(self, start_position, max_steps):
        self.position = start_position
        self.direction = Direction.N
        self.visited = set()
        self.visited.add(start_position)  # Aggiungi posizione iniziale
        self.alive = True
        self.arrived = False
        self.ahead_sight = 3
        self.lateral_sight = 1
        self.steps = 0
        self.max_steps = max_steps
        self.costs = 0
        self.fitness = None
        self.last_action = None
        self.last_inputs = []
        self.last_position = self.position
        self.stuck_counter = 0
        self.stuck = False

    # ---
    # Input processing
    # ---

    def get_inputs(self, maze: Maze):
        x, y = self.position
        s = maze.size
        # values are normalized
        return [
            x / s,
            y / s,
            self.sense_ahead(maze) / s,
            self.sense_left(maze) / s,
            self.sense_right(maze) / s
        ]

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
        return maze.first_wall(direction, *self.position, sight) or maze.size

    # ---
    # Movement
    # ---

    def turn_left(self):
        # self.costs += 0.5
        self.direction = self.direction.left

    def turn_right(self):
        # self.costs += 0.5
        self.direction = self.direction.right

    def increment_path(self, position):
        self.steps += 1
        if position not in self.visited:
            self.visited.add(position)

    def move_ahead(self, maze: Maze):
        r, c = self.position
        d = self.direction

        # checks if there's a wall
        if maze.has_wall(d, r, c):
            self.costs += 5
            return

        # moves
        self.costs += 1
        self.position = (r + self.direction.dr, c + self.direction.dc)
        self.increment_path(self.position)

    def act(self, action, maze: Maze):
        # executes an action
        match action:
            case 0:
                self.move_ahead(maze)
            case 1:
                self.turn_left()
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

        # Controlla se è bloccato
        if self.position == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > 20:
            self.stuck = True
            self.alive = False
            self.costs += 20
            return

        self.last_position = self.position

    def compute_distance_fitness(self, maze, weight):
        if self.arrived:
            self.fitness = 10000 + (1000 / (self.steps / 10))
        else:
            self.fitness = 1 / maze.distance_from_goal(self.position)
        self.fitness -= weight * self.costs
        return self.fitness



