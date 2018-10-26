import numpy as np

class grad():

    def __init__(self, val, deriv):
        self.val = val
        self.der = deriv

    def __add__(self, other):
        try:
             return grad(self.val + other.val, self.der + other.der)
        except AttributeError:
             return grad(self.val + other, self.der)

    def __radd__(self, other):
        return grad(self.val + other, self.der)

    def __sub__(self, other):
        try:
            return grad(self.val - other.val, self.der - other.der)
        except AttributeError:
            return grad(self.val - other, self.der)

    def __mul__(self, other):
        try:
            return grad(self.val * other.val, self.val * other.der + self.der * other.val)
        except AttributeError:
            return grad(self.val * other, self.der * other)

    def __rmul__(self, other):
        return grad(other * self.val, other * self.der)

    def __pow__(self, other):
        return grad(self.val**other, other * self.val**(other-1))

    def __neg__(self):
        return grad(-self.val, -self.der)
