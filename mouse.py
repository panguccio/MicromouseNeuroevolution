from maze import Maze
from maze import Direction

class Mouse:

    def __init__(self):
        self.position = [15, 0]
        self.linear_velocity = 0
        self.direction = Direction.E
        self.memorized_maze = Maze()

    def move(self):
        position = [self.position[0] + self.direction.dr, self.position[1] + self.direction.dc]

    def redirect(self, direction: Direction):
        self.direction = direction