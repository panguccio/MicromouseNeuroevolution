import maze
from direction import Direction
from maze import Maze

BONUS = 5000
maze_size = 16

# MAX CONSTANTS
max_visits = 15
max_steps = maze_size ** 2
max_stuck_counter = max_steps // 3


class Mouse:

    def __init__(self, start_position=(maze_size, 0), genome=None, gid=None, net=None,
                 generation="X"):

        # STATUS AND CHARACTERISTICS
        self.alive = True
        self.start_position = start_position
        self.position = start_position
        self.last_position = self.position
        self.direction = Direction.N
        self.arrived = False
        self.sight = 1

        # MEMORY
        self.visited_cells = set()
        self.visited_cells.add(self.start_position)
        self.closest_position = start_position

        # FOR COST COMPUTATION
        self.steps = 0
        self.collisions = 0

        # GENETICS
        self.genome = genome
        self.gid = gid
        self.net = net
        self.generation = generation

        # FITNESS
        self.fitness_values = []

    def reset(self):
        self.alive = True
        self.position = self.start_position

        self.visited_cells = set()
        self.visited_cells.add(self.start_position)
        self.closest_position = self.start_position

        self.direction = Direction.N
        self.arrived = False

        self.steps = 0
        self.collisions = 0

        self.last_position = self.position
        self.genome.fitness = 0

        if self.net is not None: self.net.reset()

    # ---
    # Input processing
    # ---

    def get_inputs(self, m: Maze):
        inputs = [
            self.sense_north(m),
            self.sense_east(m),
            self.sense_south(m),
            self.sense_west(m),
            self.relative_position_x(),
            self.relative_position_y(),
            self.proximity(m),
        ]
        return inputs

    # ---
    # Inputs
    # ---

    def sense_north(self, m: Maze):
        return self.sense(m, Direction.N, self.sight)

    def sense_east(self, m: Maze):
        return self.sense(m, Direction.E, self.sight)

    def sense_south(self, m: Maze):
        return self.sense(m, Direction.S, self.sight)

    def sense_west(self, m: Maze):
        return self.sense(m, Direction.W, self.sight)

    def sense(self, m: Maze, direction, sight):
        distance = m.first_wall(direction, *self.position, sight)
        if distance is None:
            return 0
        return 1 - distance / sight

    def proximity(self, m: Maze):
        """in which 'circle' it's positioned the mouse, that is, how close it's to the center"""
        return (maze_size // 2 - 1 - m.range_distance_from_goal(self.position)) / (maze_size // 2 - 1)

    def relative_position_x(self):
        max_distance = maze.x_distance_from_goal(self.start_position)
        return (max_distance - maze.x_distance_from_goal(self.position)) / max_distance

    def relative_position_y(self):
        max_distance = maze.y_distance_from_goal(self.start_position)
        return (max_distance - maze.y_distance_from_goal(self.position)) / max_distance

    # ---
    # Movement
    # ---

    def act(self, output, m: Maze):
        self.steps += 1
        self.genome.fitness -= 0.1

        self.direction = Direction(output)
        r, c = self.position
        self.last_position = self.position

        if m.has_wall(self.direction, r, c):
            self.genome.fitness -= 2
            self.collisions += 1
        else:
            self.position = (r + self.direction.dr, c + self.direction.dc)

        if m.range_distance_from_goal(self.position) < m.range_distance_from_goal(self.closest_position):
            self.closest_position = self.position
            self.genome.fitness += 100

        if self.position not in self.visited_cells:
            self.genome.fitness += 100
            self.visited_cells.add(self.position)
        else:
            self.genome.fitness -= 1

        if maze.is_in_goal(self.position):
            self.genome.fitness += BONUS
            self.arrived = True
            self.alive = False
            return

        if self.steps >= max_steps:
            self.genome.fitness -= 5
            self.alive = False
            return

    def explore(self, m: Maze):
        self.reset()

        while self.alive:
            inputs = self.get_inputs(m)
            outputs = self.net.activate(inputs)
            action = outputs.index(max(outputs))
            self.act(action, m)

    # ---
    # Stats for debugging
    # ---

    def stats(self):
        genetics = f"\tgeneration: {self.generation}; gid: {self.gid}\n"
        position = f"\tlast position: {self.position} -> {maze.manhattan_distance_from_goal(self.position)} from goal\n"
        fitness = f"\tfitness: {self.genome.fitness}\n"
        status = f"\tarrived? {self.arrived}.\n"
        path = f"\tsteps: {self.steps}, num visited: {len(self.visited_cells)}\n"
        costs = f"\tvisits/cell: {self.steps / len(self.visited_cells)}; coverage: {(100 * len(self.visited_cells) / (maze_size ** 2)):.2f}%; collisions: {self.collisions}\n"
        return genetics + status + position + fitness + path + costs
