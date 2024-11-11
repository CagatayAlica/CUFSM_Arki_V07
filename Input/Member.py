from typing import Literal
import numpy as np


class Member:
    def __init__(self,
                 Lx: float,
                 Ly: float,
                 Lt: float,
                 Kx: float,
                 Ky: float,
                 Kt: float,
                 support: Literal["S-S", "C-C", "S-C", "C-F", "C-G"]):
        self.Lx = Lx
        self.Ly = Ly
        self.Lt = Lt
        self.Kx = Kx
        self.Ky = Ky
        self.Kt = Kt
        self.support = support
        self.lengths_data = None
        self.lengthRange()

    def lengthRange(self):
        self.lengths_data = np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36,
            38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120,
            132, 144, 156, 168, 180, 204, 228, 252, 276, 300])
        self.lengths_data = np.sort(np.append(self.lengths_data, self.Lx))
