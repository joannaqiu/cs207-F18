import numpy as np
import matplotlib.pyplot as plt
from autodiff import autodiff as ad
from autodiff import adnp as anp

#x = ad.grad(2, 1)
N = 1000
xgrid = np.linspace(0.0, 10.0, N)
x = ad.grad(xgrid, np.ones(N))

# y = x * x * x
# y' = 3 * x * x
#y = x * x * x

# y = x^3
# y' = 3 * x^2
#y = x**3.0

# y = sin(x)
# y' = cos(x)
#y = anp.sin(x * x)

# y = e^x
# y' = e^x
#y = anp.exp(x * x)

# The example from the paper
#y = anp.exp(-anp.sqrt(x)) * anp.sin(x * anp.log(1.0 + x**2.0))
#x2 = xgrid**2.0
#arg = 1.0 + x2
#logarg = np.log(arg)
#xlogarg = xgrid * logarg
#yp = 0.5 * np.exp(-np.sqrt(xgrid)) / np.sqrt(xgrid) * (2.0 * np.sqrt(xgrid) * np.cos(xlogarg) * (2.0 * x2 / arg + logarg) - np.sin(xlogarg)) 
#
#yp_compare = np.abs(y.der[1:] - yp[1:])
#print("The largest difference in the analytical and AD derivatives is {0:25.16e}.".format(yp_compare.max()))
#print("Numerical precision on this machine is {0:25.16e}.".format(np.finfo(float).eps))

# Really messy example
y = anp.exp(-anp.sqrt(anp.log((1.0 + x)**2.0 + 1.0))) * anp.sin(x * anp.log(1.0 + x**2.0))

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(xgrid, y.val, lw=3, label=r'$y$')
ax.plot(xgrid, y.der, lw=3, label=r'$y^{\prime}$')
#ax.plot(xgrid, yp, ls='--', lw=3, label=r'$y^{\prime}_{e}$')

ax.set_xlabel(r'$x$', fontsize=22)

ax.tick_params(axis='both', which='major', labelsize=22)

ax.set_xlim(0.0, 10.0)
ax.set_ylim(-1.0, 1.0)

ax.legend(fontsize=22)

plt.show()

