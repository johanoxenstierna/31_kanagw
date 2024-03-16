import numpy as np
import matplotlib.pyplot as plt

g = np.random.poisson(1, 100) * 10
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax0 = plt.hist(g)
plt.show()

