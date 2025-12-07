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

    # ---
    # Input processing
    # ---

    def get_inputs(self, maze: Maze):
        x, y = self.position
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
        return maze.first_wall(direction, *self.position, sight)

    # ---
    # Movement
    # ---

    def turn_left(self):
        if self.alive:
            self.direction = self.direction.left
            self.steps += 0.5

    def turn_right(self):
        if self.alive:
            self.direction = self.direction.right
            self.steps += 0.5

    def move_ahead(self):
        if self.alive:
            self.position = (self.position[0] + self.direction.dr, self.position[1] + self.direction.dc)
            self.steps += 1

    def move_diagonally_left(self):
        self.move_ahead()
        self.turn_left()
        self.move_ahead()
        self.steps -= 1 # so that it costs just 0.5 more than moving ahead

    def move_diagonally_right(self):
        self.move_ahead()
        self.turn_right()
        self.move_ahead()
        self.steps -= 1

    def act(self, action, maze):
        match action:
            case 0:
                self.move_ahead()
            case 1:
                self.turn_left()
            case 2:
                self.turn_right()
            case 3:
                self.move_diagonally_left()
            case 4:
                self.move_diagonally_right()
        if self.position in maze.goal_cells:
            self.alive = False
            self.arrived = True
        if self.steps >= self.max_steps:
            self.alive = False

