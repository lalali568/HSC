import numpy as np
import matplotlib.pyplot as plt

a = np.array([0, 0, 0, 1, 1, 0, 0, 1])
figure, ax = plt.subplots()
ax.fill_between(range(len(a)), 0.8, where=(a > 0), color='blue')
plt.show()