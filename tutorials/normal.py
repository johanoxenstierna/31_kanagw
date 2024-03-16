import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
from scipy import stats

# beta_pdf = beta.pdf(x=np.linspace(0, 1, 100), a=2, b=5, loc=0)
# ax0 = plt.plot(beta_pdf)

# rvs = np.random.normal(loc=640, scale=150, size=1000)
# plt.hist(rvs, bins=100)

# _gamma = gamma.pdf(np.linspace(0, 100, 100), 2, 5, 10)
# ax0 = plt.plot(_gamma)

"""WEIGHTED LINSPACE"""
# your distribution:
distribution = stats.norm(loc=0.6, scale=0.15)

# percentile point, the range for the inverse cumulative distribution function:
bounds_for_range = distribution.cdf([0, 1])

# Linspace for the inverse cdf:
pp = np.linspace(*bounds_for_range, num=3000)

# x = distribution.ppf(pp).astype(float)
x = distribution.cdf(pp)
x[-1] = x[-2]
# And just to check that it makes sense you can try:
from matplotlib import pyplot as plt
plt.plot(x)
# plt.hist(x)
plt.show()



