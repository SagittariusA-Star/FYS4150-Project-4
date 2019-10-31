import numpy as np 
import matplotlib.pyplot as plt 


L40 = np.loadtxt("L40.txt", skiprows = 1)
spins40 = 40 ** 2
T = L40[:, 0]
E40 = L40[:, 1] / spins40
M40 = L40[:, 2] / spins40
C_v40 = L40[:, 3] / spins40
Chi40 = L40[:, 4] / spins40

fig1, ax1 = plt.subplots()
ax1.set_title(r"$\langle E \rangle$")
ax1.plot(T, E40, label = "N = 40")
ax1.legend(loc = 0)

fig2, ax2 = plt.subplots()
ax2.set_title(r"$\langle M \rangle$")
ax2.plot(T, M40, label = "N = 40")
ax2.legend(loc = 0)

fig3, ax3 = plt.subplots()
ax3.set_title(r"$C_v$")
ax3.plot(T, C_v40, label = "N = 40")
ax3.legend(loc = 0)

fig4, ax4 = plt.subplots()
ax4.set_title(r"$\chi$")
ax4.plot(T, Chi40, label = "N = 40")
ax4.legend(loc = 0)
plt.show()