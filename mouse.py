from maze import Maze
from direction import Direction


class Mouse:

    def __init__(self, start_position=[15, 0]):
        self.position = start_position
        self.direction = Direction.N
        self.visited = set()
        self.alive = True
        self.ahead_sight = 3
        self.lateral_sight = 1

    def get_inputs(self, maze):
        return [self.position, self.sense_ahead(maze), self.sense_left(maze), self.sense_right(maze)]

    def sense_ahead(self, maze):
        return self.sense(maze, self.direction, self.ahead_sight)

    def sense_left(self, maze):
        return self.sense(maze, self.direction.left, self.lateral_sight)

    def sense_right(self, maze):
        return self.sense(maze, self.direction.right, self.lateral_sight)

    def sense(self, maze, direction, sight):
        return maze.first_wall(direction, *self.position, sight)

    def move(self):
        position = [self.position[0] + self.direction.dr, self.position[1] + self.direction.dc]

    def redirect(self, direction: Direction):
        self.direction = direction