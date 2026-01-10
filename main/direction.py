from enum import Enum


class Direction(Enum):
    N, E, S, W = 0, 1, 2, 3

    @property
    def mask(self):
        """
        returns the bit mask for a certain direction
            N -> 0001
            E -> 0010
            S -> 0100
            W -> 1000
        """
        return 1 << self.value


    @property
    def dr(self):
        """returns the shift in the row after a change"""
        return [-1, 0, +1, 0][self.value]

    @property
    def dc(self):
        """returns the shift in the column after a change"""
        return [0, +1, 0, -1][self.value]

    @property
    def opposite(self):
        """returns the opposite direction"""
        return [Direction.S, Direction.W, Direction.N, Direction.E][self.value]

    @property
    def right(self):
        """returns the left direction"""
        return [Direction.E, Direction.S, Direction.W, Direction.N][self.value]

    @property
    def left(self):
        """returns the right direction"""
        return [Direction.W, Direction.N, Direction.E, Direction.S][self.value]

    @property
    def angle(self):
        """returns the angle in radians"""
        return [0, -90, 180, 90][self.value]