import numpy as np 
import matplotlib.pyplot as plt

def E_mean_analy(T):
    return - 8 * np.sinh(8 / T) / (np.cosh(8 / T) + 3)

data = np.loadtxt("E_and_M_2by2.txt", skiprows = 1)
MC = data[:, 0]
E = data[:, 1]
print(E_mean_analy(1.0), E[-1])

plt.plot(MC, E, label = r"$\langle E(MC)\rangle$")
plt.plot(MC, np.ones_like(MC) * E_mean_analy(1.0), label = r"$\langle E_{analy}(MC)\rangle$")
plt.legend(loc=0)
plt.show()