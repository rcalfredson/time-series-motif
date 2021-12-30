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
                seq = repr(v[-ord:].astype(str))
                if not seq in self.maps[ord].keys:
                    self.maps[ord][seq] = 0
                self.maps[ord][seq] += 1

    def predict(self, v):
        v = np.array(v)
        nchar = len(self.chars)
        for ord in range(self.max_ord, -1, -1,):
            scores = np.zeros((1, nchar))
            context = v[len(v) - ord + 1 :]
            for i in range(nchar):
                seq = repr(np.hstack((context, self.chars[i])).astype(str))
                if seq in self.maps[ord]:
                    scores[i] = self.maps[ord][seq]
            if np.count_nonzero(scores) > 0:
                best_char = self.chars[np.argmin(scores)]
                break
        return best_char
