import util
import matplotlib.pyplot as plt
import numpy as np
x, y = util.load_dataset('../data/ds1_valid.csv', add_intercept=False)
x[:, -1] = np.log(x[:, -1])
plt.figure()
plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)
plt.show()