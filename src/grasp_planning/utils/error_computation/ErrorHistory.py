


class ErrorHistory:
    def __init__(self):
        self.hist = []

    def reset_buffer(self):
        self.hist = []

    def append(self, error: float):
        self.hist.append(error)
