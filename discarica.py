def randomize(self):
    """randomizes the maze"""
    self.grid[:] = 0b1111

    for row in range(self.size):
        for column in range(self.size):
            random_times = random.randint(0, 2)
            for _ in range(random_times):
                d = random.choice(list(Direction))
                self.remove_wall(d, row, column)

    self.add_valid_walls()


def add_valid_walls(self):
    # create start zone
    self.remove_wall(Direction.E, *self.start_cell)
    self.add_wall(Direction.S, *self.start_cell)

    # create goal zone
    goal_internal_walls = [[Direction.E, *self.goal_cells[0]],
                           [Direction.S, *self.goal_cells[0]],
                           [Direction.S, *self.goal_cells[1]],
                           [Direction.W, *self.goal_cells[2]]]

    goal_external_walls = [[Direction.W, *self.goal_cells[0]],
                           [Direction.N, *self.goal_cells[0]],
                           [Direction.N, *self.goal_cells[1]],
                           [Direction.E, *self.goal_cells[1]],
                           [Direction.E, *self.goal_cells[2]],
                           [Direction.S, *self.goal_cells[2]],
                           [Direction.S, *self.goal_cells[3]],
                           [Direction.W, *self.goal_cells[3]]]

    self.add_walls(goal_external_walls)
    self.remove_walls(goal_internal_walls)

    # select a random wall of the goal zone to be the entrance
    goal_entrance_wall = random.choice(goal_external_walls)
    self.goal_entrance_cell = goal_entrance_wall[1:]
    self.remove_wall(*goal_entrance_wall)

    # add walls to the border of the maze
    self.add_border_walls()


def add_border_walls(self):
    """Adds a border to the maze."""
    n = self.size
    for column in range(n):
        self.add_wall(Direction.N, 0, column)
        self.add_wall(Direction.S, n - 1, column)
    for row in range(n):
        self.add_wall(Direction.W, row, 0)
        self.add_wall(Direction.E, row, n - 1)


def create_connection(self):
    furthest_cell_from_start = self.find_furthest_cell(self.start_cell, self.goal_entrance_cell)
    if furthest_cell_from_start != self.goal_entrance_cell:
        furthest_cell_from_goal = self.find_furthest_cell(self.goal_entrance_cell, furthest_cell_from_start)
        self.create_path(furthest_cell_from_start, furthest_cell_from_goal)


def find_furthest_cell(self, pointed_cell, destination_cell):
    connected_cells = set()
    best_cell = pointed_cell
    connected_cells.add(pointed_cell)
    for direction in Direction:
        if not self.has_wall(direction, *pointed_cell):
            new_pointed_cell = (pointed_cell[0] + direction.dr, pointed_cell[1] + direction.dc)
            if new_pointed_cell not in connected_cells:
                if self.distance_between(best_cell, destination_cell) > self.distance_between(new_pointed_cell,
                                                                                              destination_cell):
                    best_cell = new_pointed_cell
                self.find_furthest_cell(new_pointed_cell, destination_cell)
    return best_cell


def distance_between(self, pointed_cell, destination_cell):
    distance = 0
    distance += abs(destination_cell[0] - pointed_cell[0])
    distance += abs(destination_cell[1] - pointed_cell[1])
    return distance


# def create_path(self, cell_a, cell_b):

def generate(self, seed=None, fork_chance=0.3, turn_bias=0.7):
    """
    Generate a random maze using growing tree algorithm.
    Adapted from: https://github.com/johnnesky/rainbowmazes/blob/main/index.html
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Start with all walls
    self.grid[:] = 0b1111

    # Track visited cells
    visited = set()

    # Cell metadata: (row, col, exits)
    leaf_cells = []
    path_cells = []

    # Start from center
    start_row, start_col = self.size // 2, self.size // 2
    visited.add((start_row, start_col))
    leaf_cells.append((start_row, start_col, 0))

    # Track last direction for turn bias
    cell_last_dir = {}

    def count_unvisited_neighbors(row, col):
        """Count how many unvisited neighbors a cell has."""
        count = 0
        for direction in Direction:
            new_row = row + direction.dr
            new_col = col + direction.dc
            if self.in_bounds(new_row, new_col) and (new_row, new_col) not in visited:
                count += 1
        return count

    def carve_passage(row, col, direction):
        """Carve a passage in the given direction."""
        new_row = row + direction.dr
        new_col = col + direction.dc

        if not self.in_bounds(new_row, new_col):
            return None
        if (new_row, new_col) in visited:
            return None

        # Remove walls
        self.remove_wall(row, col, direction)
        visited.add((new_row, new_col))

        # Track last direction
        cell_last_dir[(new_row, new_col)] = direction

        return (new_row, new_col, 1)

    def try_carve_from_cell(cell_info, retry=True):
        """Try to carve from a cell using weighted direction selection."""
        row, col, exits = cell_info

        # Get available directions
        available_dirs = []
        for direction in Direction:
            new_row = row + direction.dr
            new_col = col + direction.dc
            if self.in_bounds(new_row, new_col) and (new_row, new_col) not in visited:
                available_dirs.append(direction)

        if not available_dirs:
            return None, False

        # Apply turn bias weights
        weights = []
        last_dir = cell_last_dir.get((row, col))

        for direction in available_dirs:
            weight = 1.0
            if last_dir is not None:
                # Favor continuing in same direction
                if direction == last_dir:
                    weight *= turn_bias
                else:
                    weight *= (1.0 - turn_bias)
            weights.append(weight)

        # Try directions in weighted random order
        attempts = len(available_dirs)
        for _ in range(attempts):
            if not available_dirs:
                break

            # Weighted random choice
            total = sum(weights)
            r = random.random() * total
            cumsum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    chosen_idx = i
                    break

            direction = available_dirs[chosen_idx]
            new_cell = carve_passage(row, col, direction)

            if new_cell is not None:
                return new_cell, True

            # Remove failed direction
            available_dirs.pop(chosen_idx)
            weights.pop(chosen_idx)

            if not retry:
                return None, False

        return None, False

    # Main generation loop
    cells_to_add = self.size * self.size - 1

    while cells_to_add > 0 and (leaf_cells or path_cells):
        # Decide whether to fork or extend
        fork = random.random() < fork_chance and path_cells

        if fork:
            # Pick a cell from path to fork from
            idx = random.randint(0, len(path_cells) - 1)
            cell = path_cells.pop(idx)
        else:
            # Pick from leaf cells
            if not leaf_cells:
                if not path_cells:
                    break
                fork = True
                idx = random.randint(0, len(path_cells) - 1)
                cell = path_cells.pop(idx)
            else:
                cell = leaf_cells.pop()

        # Try to carve from this cell
        new_cell, carved = try_carve_from_cell(cell, retry=True)

        if carved and new_cell:
            cells_to_add -= 1
            row, col, exits = cell
            new_row, new_col, new_exits = new_cell

            # Update exit count for original cell
            exits += 1

            # Categorize cells
            if count_unvisited_neighbors(new_row, new_col) > 0:
                leaf_cells.append(new_cell)

            if count_unvisited_neighbors(row, col) > 0:
                if exits <= 1:
                    leaf_cells.insert(0, (row, col, exits))
                elif exits >= 2:
                    path_cells.append((row, col, exits))
