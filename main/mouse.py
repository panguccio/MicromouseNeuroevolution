import maze
from direction import Direction
from maze import Maze

# Constants
ARRIVAL_BONUS = 5000
MAZE_SIZE = 16
MAX_VISITS = 15
MAX_STEPS = MAZE_SIZE ** 2
MAX_STUCK_COUNTER = MAX_STEPS // 3


class Mouse:
    """Represents a mouse agent navigating through a maze using a neural network."""

    def __init__(self, start_position=(MAZE_SIZE, 0), genome=None, gid=None, net=None,
                 generation="X"):
        # Status and characteristics
        self.alive = True
        self.start_position = start_position
        self.position = start_position
        self.last_position = self.position
        self.direction = Direction.N
        self.arrived = False
        self.sight = 1

        # Memory
        self.visited_cells = set()
        self.visited_cells.add(self.start_position)
        self.closest_position = start_position

        # Movement tracking
        self.steps = 0
        self.collisions = 0

        # Genetics
        self.genome = genome
        self.gid = gid
        self.net = net
        self.generation = generation

        # Fitness tracking
        self.fitness_values = []

    def reset(self):
        """Reset mouse to initial state for a new maze exploration."""
        self.alive = True
        self.position = self.start_position
        self.last_position = self.position
        self.direction = Direction.N
        self.arrived = False

        self.visited_cells = set()
        self.visited_cells.add(self.start_position)
        self.closest_position = self.start_position

        self.steps = 0
        self.collisions = 0
        if self.genome is not None:
            self.genome.fitness = 0

        if self.net is not None:
            self.net.reset()

    # ---
    # Inputs
    # ---

    def get_inputs(self, m: Maze):
        """Get all sensor inputs for the neural network."""
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

    def sense_north(self, m: Maze):
        return self.sense(m, Direction.N, self.sight)

    def sense_east(self, m: Maze):
        return self.sense(m, Direction.E, self.sight)

    def sense_south(self, m: Maze):
        return self.sense(m, Direction.S, self.sight)

    def sense_west(self, m: Maze):
        return self.sense(m, Direction.W, self.sight)

    def sense(self, m: Maze, direction, sight):
        """
        Sense distance to the first wall in a given direction.
        Returns normalized distance (0 = far, 1 = close).
        """
        distance = m.first_wall(direction, *self.position, sight)
        if distance is None:
            return 0
        return 1 - distance / sight

    def proximity(self, m: Maze):
        """
        Calculate proximity to goal using range distance.
        Returns normalized value where 1 = at goal, 0 = far from goal.
        """
        max_range = MAZE_SIZE // 2 - 1
        current_range = m.range_distance_from_goal(self.position)
        return (max_range - current_range) / max_range

    def relative_position_x(self):
        """Get normalized X position relative to goal (0 = start, 1 = goal)."""
        max_distance = maze.x_distance_from_goal(self.start_position)
        current_distance = maze.x_distance_from_goal(self.position)
        return (max_distance - current_distance) / max_distance

    def relative_position_y(self):
        """Get normalized Y position relative to goal (0 = start, 1 = goal)."""
        max_distance = maze.y_distance_from_goal(self.start_position)
        current_distance = maze.y_distance_from_goal(self.position)
        return (max_distance - current_distance) / max_distance

    # ---
    # Movement
    # ---

    def act(self, output, m: Maze):
        """
        Execute an action based on neural network output.
        Updates position, fitness, and checks termination conditions.
        """
        self.steps += 1
        self.genome.fitness -= 0.1

        self.direction = Direction(output)
        r, c = self.position
        self.last_position = self.position

        # Check for collision with wall
        if m.has_wall(self.direction, r, c):
            self.genome.fitness -= 2
            self.collisions += 1
        else:
            # Move to new position
            self.position = (r + self.direction.dr, c + self.direction.dc)

        # Reward getting closer to goal
        if m.range_distance_from_goal(self.position) < m.range_distance_from_goal(self.closest_position):
            self.closest_position = self.position
            self.genome.fitness += 100

        # Reward exploring new cells
        if self.position not in self.visited_cells:
            self.genome.fitness += 100
            self.visited_cells.add(self.position)
        else:
            self.genome.fitness -= 1

        # Check if goal reached
        if maze.is_in_goal(self.position):
            self.genome.fitness += ARRIVAL_BONUS
            self.arrived = True
            self.alive = False
            return

        # Check if max steps exceeded
        if self.steps >= MAX_STEPS:
            self.genome.fitness -= 5
            self.alive = False
            return

    def explore(self, m: Maze):
        """
        Explore the maze until reaching the goal or exceeding max steps.
        Uses neural network to decide actions.
        """
        self.reset()

        while self.alive:
            inputs = self.get_inputs(m)
            outputs = self.net.activate(inputs)
            action = outputs.index(max(outputs))
            self.act(action, m)

    def stats(self):
        genetics = f"\tGeneration: {self.generation}; ID: {self.gid}\n"
        status = f"\tArrived: {self.arrived}\n"
        position = (f"\tLast position: {self.position} "
                    f"-> {maze.manhattan_distance_from_goal(self.position)} from goal\n")
        fitness = f"\tFitness: {self.genome.fitness}\n"
        path = f"\tSteps: {self.steps}, Visited cells: {len(self.visited_cells)}\n"

        visits_per_cell = self.steps / len(self.visited_cells)
        coverage = 100 * len(self.visited_cells) / (MAZE_SIZE ** 2)
        costs = (f"\tVisits per cell: {visits_per_cell:.2f}; "
                 f"Coverage: {coverage:.2f}%; "
                 f"Collisions: {self.collisions}\n")

        return genetics + status + position + fitness + path + costs