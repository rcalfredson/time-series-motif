class MarkovChain:
    def __init__(self, max_ord) -> None:
        self.max_ord = max_ord
        self.maps = {}
        for i in range(max_ord):
            self.maps[i] = {}
        self.chars = []
