import numpy as np

data = np.array([4761, 4796, 4599])
mu = sum(data) / len(data)
sigma = np.std(data)
simulated_times = np.random.normal(mu, sigma, size=15)
for d in data:
    print(round(d, 0), end=',')
for si in simulated_times:
    print(int(si), end=',')

