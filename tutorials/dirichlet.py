import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


aa = np.random.random_integers(4, 20, size=50)

gamma = stats.gamma.pdf(np.linspace(0, 100, 100), 15, 20)

''' time between fires'''
_dirichlet = np.random.dirichlet((200, 170, 180), 20)

ax0 = plt.hist(_dirichlet[:, 0])
# ax0 = plt.hist(ff, bins=100)
plt.show()

df = 5


