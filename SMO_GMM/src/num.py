import math
import ast

class NUM:
    def __init__(self,s=None,n=None):
        self.txt = s or " "
        self.at = n or 0
        self.n=0
        self.mu=0
        self.m2=0
        self.hi = -float('inf')
        self.low = float('inf')
        self.heaven = 0 if (s or "").endswith("-") else 1

    def add(self,x,d=0):
        if x != "?" and x.replace(".", "", 1).isdigit():
            x = ast.literal_eval(x)
            self.n+=1
            d = x - self.mu
            self.mu += d/self.n
            self.m2 += d*(x-self.mu)
            self.low = min(x,self.low)
            self.hi = max(x,self.hi)

    def mid(self):
        return self.mu
    
    def div(self):
        return 0 if self.n < 2 else (self.m2 / (self.n - 1))**0.5
    
    def small(self):
        pass

    def norm(self, x):
        if x == "?":
            return x
        else:
            return (x - self.low) / (self.hi - self.low + 1E-30)

    def like(self, x, _):
        mu, sd = self.mid(), (self.div() + 1E-30)
        try:
            x_float = float(x)
        except ValueError:
            return 0.0
        nom = math.exp(-0.5 * ((float(x) - mu) ** 2) / (sd ** 2))
        denom = (sd * 2.5 + 1E-30)
        return nom / denom
