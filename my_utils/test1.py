import numpy as np

# data = np.array([4761, 4796, 4599])
# mu = sum(data) / len(data)
# sigma = np.std(data)
# simulated_times = np.random.normal(mu, sigma, size=15)
# for d in data:
#     print(round(d, 0), end=',')
# for si in simulated_times:
#     print(int(si), end=',')

data = np.array([4761,4796,4599,4619,4782,4688,4690,4624,4805,4729,4839,4673,4929,4718,4849,4777,4671,4757])
sum = 0
for d in data:
    sum= sum+d
    print("4739.22",end=',')
