import numpy as np
import matplotlib.pyplot as plt

# Setting fonts for pretty plot
fonts = {
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(fonts)


L40 = np.loadtxt("L40.txt", skiprows=1)
L60 = np.loadtxt("L60.txt", skiprows=1)
L80 = np.loadtxt("L80.txt", skiprows=1)
L100 = np.loadtxt("L100.txt", skiprows=1)

T = L40[:, 0]
N40 = 40 ** 2
E40 = L40[:, 1] / N40
M40 = L40[:, 2] / N40
C_v40 = L40[:, 3] / N40
Chi40 = L40[:, 4] / N40

N60 = 60 ** 2
E60 = L60[:, 1] / N60
M60 = L60[:, 2] / N60
C_v60 = L60[:, 3] / N60
Chi60 = L60[:, 4] / N60

N80 = 80 ** 2
E80 = L80[:, 1] / N80
M80 = L80[:, 2] / N80
C_v80 = L80[:, 3] / N80
Chi80 = L80[:, 4] / N80

N100 = 100 ** 2
E100 = L100[:, 1] / N100
M100 = L100[:, 2] / N100
C_v100 = L100[:, 3] / N100
Chi100 = L100[:, 4] / N100


fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].scatter(T, E40, label="N = 40", color="r", s=10)
ax[0, 0].scatter(T, E60, label="N = 60", color="b", s=10)
ax[0, 0].scatter(T, E80, label="N = 80", color="g", s=10)
ax[0, 0].scatter(T, E100, label="N = 100", color="orange", s=10)
ax[0, 0].legend(loc=0)
ax[0, 0].set_ylabel(r"$\frac{\langle E(T) \rangle}{N^2}$")
# ax[0, 0].set_xlabel(r"$T$")
ax[0, 0].grid()

ax[1, 0].scatter(T, M40, label="N = 40", color="r", s=10)
ax[1, 0].scatter(T, M60, label="N = 60", color="b", s=10)
ax[1, 0].scatter(T, M80, label="N = 80", color="g", s=10)
ax[1, 0].scatter(T, M100, label="N = 100", color="orange", s=10)
ax[1, 0].legend(loc=0)
ax[1, 0].set_ylabel(r"$\frac{\langle |M(T)| \rangle}{N^2}$")
ax[1, 0].set_xlabel(r"$T$")
ax[1, 0].grid()

ax[0, 1].scatter(T, C_v40, label="N = 40", color="r", s=10)
ax[0, 1].scatter(T, C_v60, label="N = 60", color="b", s=10)
ax[0, 1].scatter(T, C_v80, label="N = 80", color="g", s=10)
ax[0, 1].scatter(T, C_v100, label="N = 100", color="orange", s=10)
ax[0, 1].legend(loc=0)
ax[0, 1].set_ylabel(r"$\frac{C_v(T)}{N^2}$")
# ax[0, 1].set_xlabel(r"$T$")
ax[0, 1].grid()

ax[1, 1].scatter(T, Chi40, label="N = 40", color="r", s=10)
ax[1, 1].scatter(T, Chi60, label="N = 60", color="b", s=10)
ax[1, 1].scatter(T, Chi80, label="N = 80", color="g", s=10)
ax[1, 1].scatter(T, Chi100, label="N = 100", color="orange", s=10)
ax[1, 1].legend(loc=0)
ax[1, 1].set_ylabel(r"$\frac{\chi(T)}{N^2}$")
ax[1, 1].set_xlabel(r"$T$")
ax[1, 1].grid()

fig.tight_layout(w_pad=1)
fig.savefig("../doc/Figures/thermo_quants.pdf", dpi=1000)
plt.show()

# Computing critical temperature
L100 = 100
L80 = 80
nu = 1
TC100 = T[np.argmax(C_v100)]
TC80 = T[np.argmax(C_v80)]
a = (TC100 - TC80) / (L100 ** (-1 / nu) - L80 ** (-1 / nu))
TC = TC100 - a * L100 ** (-1 / nu)
TC_analytical = 2.269
epsilon = abs(TC - TC_analytical) / TC_analytical
print("Critical Temperature: ", TC, " Relative error: ", epsilon)
