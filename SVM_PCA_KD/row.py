import math
from constants import *
from helpers import *


class ROW:
    def __init__(self, t):
        self.cells = t
        # self.k = constants.the['k']

    @classmethod
    def new(cls, t):
        return cls(t)

    def like(self, data, n, nHypotheses):
        prior = (len(data.row) + the['k']) / (n + the['k'] * nHypotheses)
        out = math.log(prior) if prior != 0 else float("-inf")
        for col in data.cols.x.values():
            v = self.cells[col.at]
            if v != "?":
                inc = col.like(v, prior)
                out = out + (math.log(inc) if inc != 0 else float("-inf"))
        return math.exp(1) ** out

    def likes(self, datas):
        n, nHypotheses = 0, 0
        most, out = None, None

        for k, data in datas.items():
            n += len(data.row)
            nHypotheses += 1

        for k, data in datas.items():
            tmp = self.like(data, n, nHypotheses)
            if most is None or tmp > most:
                most, out = tmp, k

        return out, most
    
    def d2h(self, data):
        d, n = 0, 0

        for col in data.cols.y.values():
            n += 1
            norm = col.norm(coerce(self.cells[col.at]))
            d += abs(col.heaven - norm) ** 2

        return (d ** 0.5) / (n ** 0.5)

