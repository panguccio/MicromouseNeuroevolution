import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import random

from maze import Maze


class MazeLoader:
    def __init__(self):
        self.directory = "./mazes"
        self.api_url = "https://api.github.com/repos/micromouseonline/mazefiles/git/trees/master?recursive=1"
        self.base_url = "https://raw.githubusercontent.com/micromouseonline/mazefiles/master/classic/"
        self.maze_names = []
        self.load_mazes()

    def load_mazes(self):

        if not os.path.exists(self.directory):

            res = requests.get(self.api_url).json()

            self.maze_names = [
                file["path"].split("/")[1] for file in res["tree"]
                if file["path"].startswith("classic/") and file["path"].endswith(".txt")
            ]

            session = requests.Session()

            os.makedirs(self.directory, exist_ok=True)

            def download_file(name):
                path = os.path.join(self.directory, name)
                text = session.get(self.base_url + name).text
                text = self.fix_maze_content(text)
                if len(text) > 15:
                    with open(path, "w") as f:
                        f.write(text)
                else:
                    print(name)

            with ThreadPoolExecutor(max_workers=10) as pool:
                list(tqdm(pool.map(download_file, self.maze_names),
                          total=len(self.maze_names),
                          desc="Downloading mazes"))
        else:
            self.maze_names = [
                file for file in os.listdir(os.path.join(self.directory))
            ]

    def fix_maze_content(self, text):
        """Some mazes have wrong characters, to make them uniform in representation this method replaces them."""
        mapping = {
            '.': 'o',
            'G': ' ',
            'S': ' ',
            'A': ' '
        }
        return ''.join(mapping.get(c, c) for c in text)

    def get_random_maze(self):
        name = random.choice(self.maze_names)
        return self.get_maze(name)

    def get_maze(self, name):
        f = open(self.directory + "/" + name, "r")
        return Maze(text=f.readlines(), name=name)

    def get_random_mazes(self, quantity=10):
        mazes = []
        for i in range(quantity):
            mazes.append(self.get_random_maze())
        return mazes

    def get_all_mazes(self):
        return self.maze_names