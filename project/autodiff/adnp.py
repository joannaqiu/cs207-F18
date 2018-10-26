import numpy as np
from autodiff import autodiff as _ad

def sin(x):
    return _ad.grad(np.sin(x.val), np.cos(x.val) * x.der)

def cos(x):
    return _ad.grad(np.cos(x.val), -np.sin(x.val) * x.der)

def exp(x):
    return _ad.grad(np.exp(x.val), np.exp(x.val) * x.der)

def log(x):
    return _ad.grad(np.log(x.val), x.der / x.val)

def log10(x):
    return _ad.grad(np.log10(x.val), x.der / x.val / np.log(10))

def log2(x):
    return _ad.grad(np.log10(x.val), x.der / x.val / np.log(2))

def sqrt(x):
    return _ad.grad(np.sqrt(x.val), 0.5 * x.der / np.sqrt(x.val))
