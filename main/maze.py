import numpy as np

from main.direction import Direction

# Constants
SIZE = 16
MID = SIZE // 2
GOAL_CELLS = [
    (MID - 1, MID - 1),  # top-left
    (MID - 1, MID),  # top-right
    (MID, MID),  # bottom-right
    (MID, MID - 1)  # bottom-left
]
START_CELL = (15, 0)


def manhattan_distance_from_goal(pointed_cell):
    """Calculate minimum Manhattan distance from a cell to any goal cell."""
    return min(manhattan(pointed_cell, goal_cell) for goal_cell in GOAL_CELLS)


def is_in_goal(pointed_cell):
    """Check if a cell is in the goal area."""
    return manhattan_distance_from_goal(pointed_cell) == 0


def x_distance_from_goal(pointed_cell):
    """Calculate minimum horizontal distance from a cell to any goal cell."""
    return min(abs(goal_cell[1] - pointed_cell[1]) for goal_cell in GOAL_CELLS)


def y_distance_from_goal(pointed_cell):
    """Calculate minimum vertical distance from a cell to any goal cell."""
    return min(abs(goal_cell[0] - pointed_cell[0]) for goal_cell in GOAL_CELLS)


def manhattan(cell_a, cell_b):
    """Calculate Manhattan distance between two cells."""
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])


class Maze:
    """Represents a maze grid with walls and visit tracking."""

    def __init__(self, text=None, name=""):
        self.size = SIZE
        self.grid = np.zeros((SIZE, SIZE), dtype=np.uint8)
        self.name = name

        if text is not None:
            self._from_text(text)

    def _from_text(self, text):
        """Parse maze from text representation."""
        rows = SIZE * 2 + 1
        columns = SIZE * 4 + 1

        for row in range(rows):
            for column in range(columns):
                token = text[row][column]

                if token == " ":
                    column += 3
                elif token == "-":
                    direction = Direction.N
                    cell = (row // 2, column // 4)
                    self.add_wall(direction, *cell)
                    column += 3
                elif token == "|":
                    direction = Direction.W
                    cell = (row // 2, column // 4)
                    self.add_wall(direction, *cell)

    def in_bounds(self, row, column):
        """Check if a cell is within maze bounds."""
        return 0 <= row < self.size and 0 <= column < self.size

    def get_walls(self, row, column):
        """Get wall bitmask for a cell (lower 4 bits)."""
        return self.grid[row, column] & 15

    def has_wall(self, direction: Direction, row, column):
        """Check if a cell has a wall in the specified direction."""
        return bool(self.grid[row, column] & direction.mask)

    def add_wall(self, direction: Direction, row, column):
        """Add a wall to a cell in the specified direction."""
        self._add_cell_wall(row, column, direction)

        # Update adjacent cell
        new_row = row + direction.dr
        new_column = column + direction.dc
        opposite_direction = direction.opposite
        self._add_cell_wall(new_row, new_column, opposite_direction)

    def _add_cell_wall(self, row, column, direction):
        """Add a wall to a specific cell if in bounds."""
        if self.in_bounds(row, column):
            self.grid[row, column] |= direction.mask

    def add_walls(self, walls):
        """Add multiple walls from a list of (direction, row, column) tuples."""
        for wall in walls:
            direction, cell = wall[0], wall[1:]
            self.add_wall(direction, *cell)

    def remove_wall(self, direction: Direction, row, column):
        """Remove a wall from a cell in the specified direction."""
        if not self.in_bounds(row, column):
            return

        self._remove_cell_wall(row, column, direction)

        # Update adjacent cell
        new_row = row + direction.dr
        new_column = column + direction.dc
        opposite_direction = direction.opposite
        self._remove_cell_wall(new_row, new_column, opposite_direction)

    def _remove_cell_wall(self, row, column, direction):
        """Remove a wall from a specific cell if in bounds."""
        if self.in_bounds(row, column):
            self.grid[row, column] &= np.invert(direction.mask) | 240

    def remove_walls(self, walls):
        """Remove multiple walls from a list of (direction, row, column) tuples."""
        for wall in walls:
            direction, cell = wall[0], wall[1:]
            self.remove_wall(direction, *cell)

    def add_visit(self, row, column):
        """Increment visit counter for a cell (max 15 visits)."""
        if not self.in_bounds(row, column):
            return

        times_visited = self.get_visits(row, column)
        if times_visited == 15:
            return

        times_visited += 1
        self.grid[row, column] = (times_visited << 4) | self.get_walls(row, column)

    def get_visits(self, row, column):
        """Get number of visits for a cell (upper 4 bits)."""
        return self.grid[row, column] >> 4 if self.in_bounds(row, column) else 0

    def first_wall(self, direction: Direction, row, column, max_depth=16):
        """
        Find distance to first wall in a direction.
        Returns the number of steps to the first wall, or None if no wall found.
        """
        for step in range(max_depth):
            current_row = row + (direction.dr * step)
            current_column = column + (direction.dc * step)
            cell = (current_row, current_column)

            if self.in_bounds(*cell):
                if self.has_wall(direction, *cell):
                    return step

        return None

    def range_distance_from_goal(self, pointed_cell):
        """Calculate range distance from a cell to the goal."""
        return self._range_distance(pointed_cell)

    def _range_distance(self, cell):
        """
        Calculate range distance (concentric square distance) from center.
        Returns 0 for center cells, increasing outward.
        """
        for i in range(self.size // 2):
            if min(cell) == i or max(cell) == self.size - 1 - i:
                return self.size // 2 - 1 - i

        return 0

    def print_grid(self):
        """Print ASCII representation of the maze with visit counts."""
        for r in range(SIZE):
            # Print top walls
            line = "+"
            for c in range(SIZE):
                line += "---+" if self.has_wall(Direction.N, r, c) else "   +"
            print(line)

            # Print side walls and visit counts
            line = ""
            for c in range(SIZE):
                line += "|" if self.has_wall(Direction.W, r, c) else " "
                line += f"{self.get_visits(r, c):3d}"
            line += "|" if self.has_wall(Direction.E, r, SIZE - 1) else " "
            print(line)

        # Print bottom walls
        line = "+"
        for c in range(SIZE):
            line += "---+" if self.has_wall(Direction.S, SIZE - 1, c) else "   +"
        print(line)

    def print_grid_values(self):
        """Print raw grid values (for debugging)."""
        print(self.grid)