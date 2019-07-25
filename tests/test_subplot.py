
import numpy as np
from matplotlib import pyplot

pyplot.subplot(1,2,1)

x = np.arange(10.)
y = np.arange(10.)
pyplot.plot(x,y)

pyplot.subplot(1,2,2)

x = np.arange(10.)
y = np.arange(10.)
pyplot.plot(x,y)

ax = pyplot.gca()


ax = pyplot.gca()

pyplot.text(0.5,0.5, 'debug')

pyplot.savefig('debug.png')

