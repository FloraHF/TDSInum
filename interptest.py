import numpy as np
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xvals_1 = np.linspace(0, 2*np.pi, 50)
xvals_2 = np.linspace(0, 2*np.pi, 50)
yinterp = np.interp([xvals_1, xvals_2], [x]*2, [y]*2)
import matplotlib.pyplot as plt
plt.plot(yinterp[0], yinterp, '-x')
plt.show()