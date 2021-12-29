import numpy as np


class MarkovChain:
    def __init__(self, max_ord) -> None:
        self.max_ord = max_ord
        self.maps = {}
        for i in range(max_ord):
            self.maps[i] = {}
        self.chars = []

    def update(self, v):
        v = np.array(v)
        for ord in range(self.max_ord):
            if len(v) > ord:
                if v[-1] not in self.chars:
                    self.chars.append(v[-1])
                seq = v[-ord:].astype(str)
                if not any([el in self.maps[ord].keys for el in seq]):
                    self.maps[ord][]
