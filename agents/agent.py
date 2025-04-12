import torch
import numpy as np

class Agent:
    def __init__(self, idx):
        self.n_lstm = 64
        self._reset()
        self.agent_id = idx

    def _reset(self):
        # forget the cumulative states every cum_step
        self.hidden = torch.from_numpy(np.zeros((2, self.n_lstm * 2), dtype=np.float32))
        self.message = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))

    def update(self, hidden, message):
        self.hidden = hidden
        self.message = message