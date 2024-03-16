import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
from src.trig_functions import min_max_normalization



beta_pdf = beta.pdf(x=np.linspace(0, 1, 100), a=2, b=5, loc=0)
beta_pdf = min_max_normalization(beta_pdf, y_range=[1, 25])
ax0 = plt.plot(beta_pdf)

# beta_rvs = beta.rvs(a=2, b=5, loc=0, scale=200, size=25000)
# plt.hist(beta_rvs, bins=100)

# _gamma = gamma.pdf(np.linspace(0, 100, 100), 2, 5, 10)
# ax0 = plt.plot(_gamma)

plt.show()





