from maze import Maze
from direction import Direction


class Mouse:

    def __init__(self, start_position, max_steps):
        self.position = start_position
        self.direction = Direction.N
        self.visited = set()
        self.alive = True
        self.arrived = False
        self.ahead_sight = 3
        self.lateral_sight = 1
        self.steps = 0
        self.max_steps = max_steps
        self.costs = 0

    # ---
    # Input processing
    # ---

    def get_inputs(self, maze: Maze):
        x, y = self.position
        s = maze.size
        return [x, y, self.sense_ahead(maze), self.sense_left(maze), self.sense_right(maze)]

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
        self.costs += 1
        self.direction = self.direction.left


    def turn_right(self):
        self.costs += 1
        self.direction = self.direction.right

    def increment_path(self, position):
        self.steps += 1
        if position not in self.visited:
            self.visited.add(position)


    def move_ahead(self, maze: Maze):
        self.costs += 1
        r, c = self.position
        d = self.direction
        if maze.has_wall(d, r, c):
            self.costs += 5
            return
        self.position = (self.position[0] + self.direction.dr, self.position[1] + self.direction.dc)
        self.increment_path(self.position)

    """
    def move_diagonally_left(self, maze: Maze):
        r, c = self.position
        d = self.direction
        if maze.has_wall(d, r, c):
            self.costs += 5
            return
        self.steps += 1.5
        self.move_ahead(maze)
        self.turn_left()
        self.move_ahead(maze)

    def move_diagonally_right(self, maze: Maze):
        r, c = self.position
        d = self.direction
        if maze.has_wall(d, r, c):
            self.costs += 5
            return
        self.steps += 1.5
        self.move_ahead(maze)
        self.turn_right()
        self.move_ahead(maze)
        self.steps -= 2.5
    """
    def act(self, action, maze: Maze):
        match action:
            case 0:
                self.move_ahead(maze)
            case 1:
                self.turn_left()
            case 2:
                self.turn_right()
        if maze.is_in_goal(self.position):
            self.alive = False
            self.arrived = True
        if self.steps >= self.max_steps:
            self.alive = False

    def compute_fitness(self, maze, novelty_score, a=-0.53, b=0.96):
        distance = maze.distance_from_goal(self.position)
        if self.arrived:
            return 1000 + (1000 / 1 + self.steps)
        return a * 1/distance + b * novelty_score


