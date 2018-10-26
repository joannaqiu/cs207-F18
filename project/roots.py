from autodiff import autodiff as ad
from autodiff import adnp as anp

def f(a):
    x = ad.grad(a, 1.0)
    return anp.exp(-anp.sqrt(x)) * anp.sin(x * anp.log(1.0 + x**2.0))

x = 5.0
tol = 1.0e-06
nmax = 25
dx = 1.0
for nli in range(nmax):
    evalf = f(x)
    dx = -evalf.val / evalf.der
    x = x + dx
    print("{0}    {1:8.6f}     {2:8.6e}".format(nli+1, x, dx))
    if abs(dx) <= tol:
       break

print("There is a root at x = {0:6.4f}.".format(x))
