import os
import random
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

from main.maze import Maze


class MazeLoader:
    """Handles loading and downloading maze files from a remote repository."""

    MAZES_DIRECTORY = "./mazes"
    API_URL = "https://api.github.com/repos/micromouseonline/mazefiles/git/trees/master?recursive=1"
    BASE_URL = "https://raw.githubusercontent.com/micromouseonline/mazefiles/master/classic/"
    MAX_WORKERS = 10

    def __init__(self):
        self.directory = self.MAZES_DIRECTORY
        self.maze_names = []
        self.load_mazes()

    def load_mazes(self):
        """Load mazes from local directory or download from repository if not present."""
        if not os.path.exists(self.directory):
            self._download_mazes_from_repository()
        else:
            self.maze_names = [
                file for file in os.listdir(self.directory)
                if file.endswith('.txt')
            ]

    def _download_mazes_from_repository(self):
        """Download all maze files from the GitHub repository."""
        response = requests.get(self.API_URL).json()

        self.maze_names = [
            file["path"].split("/")[1]
            for file in response["tree"]
            if file["path"].startswith("classic/") and file["path"].endswith(".txt")
        ]

        os.makedirs(self.directory, exist_ok=True)

        session = requests.Session()

        def download_file(name):
            """Download and save a single maze file."""
            path = os.path.join(self.directory, name)
            text = session.get(self.BASE_URL + name).text
            text = self._fix_maze_content(text)

            if len(text) > 15:
                with open(path, "w") as f:
                    f.write(text)
            else:
                print(f"Warning: Skipped {name} (file too small)")

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            list(tqdm(
                pool.map(download_file, self.maze_names),
                total=len(self.maze_names),
                desc="Downloading mazes"
            ))

    def _fix_maze_content(self, text):
        """
        Normalize maze content by replacing non-standard characters.
        Some mazes use different characters for the same purpose.
        """
        mapping = {
            '.': 'o',
            'G': ' ',
            'S': ' ',
            'A': ' '
        }
        return ''.join(mapping.get(c, c) for c in text)

    def get_maze(self, name):
        """Load and return a specific maze by name."""
        maze_path = os.path.join(self.directory, name)
        with open(maze_path, "r") as f:
            return Maze(text=f.readlines(), name=name)

    def get_random_maze(self):
        """Load and return a random maze from available mazes."""
        name = random.choice(self.maze_names)
        return self.get_maze(name)

    def get_random_mazes(self, quantity=10):
        """Load and return multiple random mazes."""
        return [self.get_random_maze() for _ in range(quantity)]

    def get_all_maze_names(self):
        """Return list of all available maze names."""
        return self.maze_names