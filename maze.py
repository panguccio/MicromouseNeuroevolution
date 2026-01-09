import numpy as np
from direction import Direction


class Maze:

    def __init__(self, text=None, name="", size=16):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.start_cell = (15, 0)
        self.name = name
        self.max_steps = 2 * size ** 2
        mid = size // 2
        self.goal_cells = [
            (mid - 1, mid - 1),  # top-left
            (mid - 1, mid),  # top-right
            (mid, mid),  # bottom-right
            (mid, mid - 1)  # bottom-left
        ]
        self.destination = ()
        if text is not None:
            self._from_text(text, size)

    def _from_text(self, text, size):
        rows = size * 2 + 1
        columns = size * 4 + 1
        for row in range(rows):
            for column in range(columns):
                token = text[row][column]
                match token:
                    case " ":
                        column += 3
                    case "-":
                        direction = Direction.N
                        cell = (row // 2, column // 4)
                        self.add_wall(direction, *cell)
                        column += 3
                        pass
                    case "|":
                        direction = Direction.W
                        cell = (row // 2, column // 4)
                        self.add_wall(direction, *cell)
                        pass
        self.find_destination()

    def find_destination(self):
        directions = [[Direction.W, Direction.N], [Direction.N, Direction.E], [Direction.E, Direction.S], [Direction.S, Direction.W]]
        for i, (r, c) in enumerate(self.goal_cells):
            for d in directions[i]:
                if not self.has_wall(d, r, c):
                    self.destination = (r + d.dr, c + d.dc)




    def in_bounds(self, row, column):
        """checks if a cell is in maze bounds"""
        return 0 <= row < self.size and 0 <= column < self.size

    def get_walls(self, row, column):
        return self.grid[row, column] & 15

    # se è out of bounds row e column?
    def has_wall(self, direction: Direction, row, column):
        """checks if a cell has a wall in that direction"""
        return bool(self.grid[row, column] & direction.mask)

    def add_wall(self, direction: Direction, row, column):
        """adds a wall to a cell in that direction"""
        self.add_cell_wall(column, direction, row)

        # updates adjacent cell
        new_row, new_column = row + direction.dr, column + direction.dc
        new_side = direction.opposite
        self.add_cell_wall(new_column, new_side, new_row)

    def add_cell_wall(self, column, direction, row):
        if self.in_bounds(row, column):
            self.grid[row, column] |= direction.mask

    def add_walls(self, walls):
        for wall in walls:
            direction, cell = wall[0], wall[1:]
            self.add_wall(direction, *cell)

    def remove_wall(self, direction: Direction, row, column):
        """removes a wall from a cell in that direction"""
        if not self.in_bounds(row, column):
            return
        self.remove_cell_wall(column, direction, row)

        # updates adjacent cell
        new_row, new_column = row + direction.dr, column + direction.dc
        new_direction = direction.opposite
        self.remove_cell_wall(new_column, new_direction, new_row)

    def remove_cell_wall(self, column, direction, row):
        if self.in_bounds(row, column):
            self.grid[row, column] &= np.invert(direction.mask) | 240

    def remove_walls(self, walls):
        for wall in walls:
            direction, cell = wall[0], wall[1:]
            self.remove_wall(direction, *cell)

    def add_visit(self, row, column):
        if self.in_bounds(row, column):
            times_visited = self.get_visits(row, column)
            if times_visited == 15:
                return
            times_visited += 1
            self.grid[row, column] = times_visited << 4 | self.get_walls(row, column)

    def get_visits(self, row, column):
        return self.grid[row, column] >> 4 if self.in_bounds(row, column) else 0

    def print_grid(self):
        """prints the grid"""
        size = self.size
        for r in range(size):
            line = "+"
            for c in range(size):
                line += "---+" if self.has_wall(Direction.N, r, c) else "   +"
            print(line)
            line = ""
            for c in range(size):
                line += "|" if self.has_wall(Direction.W, r, c) else " "
                line += f"{self.get_visits(r, c):3d}"
            line += "|" if self.has_wall(Direction.E, r, size - 1) else " "
            print(line)
        line = "+"
        for c in range(size):
            line += "---+" if self.has_wall(Direction.S, size - 1, c) else "   +"
        print(line)

    def print_grid_values(self):
        """prints the grid values"""
        print(self.grid)

    def first_wall(self, direction: Direction, row, column, max_depth=16):
        for step in range(max_depth):
            cell = row + (direction.dr * step), column + (direction.dc * step)
            if self.in_bounds(*cell):
                if self.has_wall(direction, *cell):
                    return step

    def manhattan_distance_from_gate(self, pointed_cell):
        return manhattan(pointed_cell, self.destination)

    def manhattan_distance_from_goal(self, pointed_cell):
        return min(manhattan(pointed_cell, goal_cell) for goal_cell in self.goal_cells)

    def x_distance_from_goal(self, pointed_cell):
        return abs(self.destination[1] - pointed_cell[1])

    def y_distance_from_goal(self, pointed_cell):
        return abs(self.destination[0] - pointed_cell[0])

    def range_distance_from_goal(self, pointed_cell):
        return self.range_distance(pointed_cell)

    def range_distance(self, cell_a):
        for i in range(self.size // 2): # 0 to 8
            if min(cell_a) == i or max(cell_a) == self.size - 1 - i:
                return self.size // 2 - 1 - i

    def is_in_goal(self, pointed_cell):
        return self.manhattan_distance_from_goal(pointed_cell) == 0

def manhattan(cell_a, cell_b):
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])

