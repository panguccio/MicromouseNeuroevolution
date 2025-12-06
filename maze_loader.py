import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import random

from maze import Maze


class MazeLoader:
    def __init__(self):
        self.base_url = "https://www.tcp4me.com/mmr/mazes/"
        self.maze_names = []
        self.load_mazes()

    def load_mazes(self):
        r = requests.get(self.base_url)
        session = requests.Session()
        soup = BeautifulSoup(r.text, "html.parser")

        self.maze_names = [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].endswith(".maze")
        ]

        self.directory = "./mazes"

        if not os.path.exists(self.directory):

            os.makedirs(self.directory, exist_ok=True)

            def download_file(name):
                path = os.path.join(self.directory, name)
                text = session.get(self.base_url + name).text
                text = self.fix_maze_content(text)
                with open(path, "w") as f:
                    f.write(text)

            with ThreadPoolExecutor(max_workers=10) as pool:
                list(tqdm(pool.map(download_file, self.maze_names),
                          total=len(self.maze_names),
                          desc="Downloading mazes"))



    def fix_maze_content(self, text):
        """Some mazes have wrong characters, to make them uniform in representation this method replaces them."""
        mapping = {
            '.': '+',
            'G': '+',
            'S': ' ',
            'A': '+'
        }
        return ''.join(mapping.get(c, c) for c in text)

    def get_random_maze(self):
        name = random.choice(self.maze_names)
        return self.get_maze(name)

    def get_maze(self, name):
        f = open(self.directory + "/" + name, "r")
        return Maze(f.readlines())

    def get_all_mazes(self):
        return self.maze_names