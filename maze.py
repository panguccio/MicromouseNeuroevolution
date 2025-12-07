import numpy as np
from direction import Direction


class Maze:

    def __init__(self, text=None, size=16):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.start_cell = (15, 0)
        self.max_steps = 2 * size ** 2
        mid = size // 2
        self.goal_cells = [
            (mid - 1, mid - 1),  # top-left
            (mid - 1, mid),  # top-right
            (mid, mid),  # bottom-right
            (mid, mid - 1)  # bottom-left
        ]

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

    def in_bounds(self, row, column):
        """checks if a cell is in maze bounds"""
        return 0 <= row < self.size and 0 <= column < self.size

    def get_walls(self, row, column):
        """returns entire grid"""
        return self.grid[row, column]

    # se è out of bounds row e column?
    def has_wall(self, direction: Direction, row, column):
        """checks if a cell has a wall in that direction"""
        return bool(self.grid[row, column] & direction.mask)

    def add_wall(self, direction: Direction, row, column):
        """adds a wall to a cell in that direction"""
        if self.in_bounds(row, column):
            self.grid[row, column] |= direction.mask

        # updates adjacent cell
        new_row, new_column = row + direction.dr, column + direction.dc
        if self.in_bounds(new_row, new_column):
            new_side = direction.opposite
            self.grid[new_row, new_column] |= new_side.mask

    def add_walls(self, walls):
        for wall in walls:
            self.add_wall(wall[0], *wall[1:])

    def remove_wall(self, direction: Direction, row, column):
        """removes a wall from a cell in that direction"""
        if not self.in_bounds(row, column):
            return
        self.grid[row, column] &= np.invert(direction.mask)

        # updates adjacent cell
        new_row, new_column = row + direction.dr, column + direction.dc
        if self.in_bounds(new_row, new_column):
            new_direction = direction.opposite
            self.grid[new_row, new_column] &= np.invert(new_direction.mask)

    def remove_walls(self, walls):
        for wall in walls:
            self.remove_wall(wall[0], *wall[1:])

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
                line += "   "
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

    def distance_from_goal(self, pointed_cell):
        # the manhattan distance from the closest goal cell
        return min(sum(abs(coord1 - coord2) for coord1, coord2 in zip(pointed_cell, goal_cell)) for goal_cell in
                   self.goal_cells)
