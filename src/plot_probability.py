import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# Using numba because this was very slow.
@nb.njit
def fast_cumsum(arr, MC, norm_points):
    return np.cumsum(arr) / (MC * norm_points)

def read_file_fast(filename):
    return np.load(filename)

T1_Dis_data = read_file_fast("P_T1_MC1e7_Disordered.npy")# np.loadtxt("P_T1_MC1e7_Disordered.txt", skiprows=2)
T1_Ord_data = read_file_fast("P_T1_MC1e7_Ordered.npy")
T24_Dis_data = read_file_fast("P_T24_MC1e7_Disordered.npy")
T24_Ord_data = read_file_fast("P_T24_MC1e7_Ordered.npy")
E1_Dis = T1_Dis_data[:, 0]
E1_Ord = T1_Ord_data[:, 0]
E24_Dis = T24_Dis_data[:, 0]
E24_Ord = T24_Ord_data[:, 0]

M1_Dis = T1_Dis_data[ :, 1]
M1_Ord = T1_Ord_data[ :, 1]
M24_Dis = T24_Dis_data[ :, 1]
M24_Ord = T24_Ord_data[ :, 1]

flip1_Dis = T1_Dis_data[ :, 2]
flip1_Ord = T1_Ord_data[ :, 2]
flip24_Dis = T24_Dis_data[ :, 2]
flip24_Ord = T24_Ord_data[ :, 2]

MC = np.arange(1, len(E1_Dis) + 1, 1)
E1_Dis_mean = fast_cumsum(E1_Dis, MC, 400)
E1_Ord_mean = fast_cumsum(E1_Ord, MC, 400)
E24_Dis_mean = fast_cumsum(E24_Dis, MC, 400)
E24_Ord_mean = fast_cumsum(E24_Ord, MC, 400)

M1_Dis_mean = fast_cumsum(M1_Dis, MC, 400)
M1_Ord_mean = fast_cumsum(M1_Ord, MC, 400)
M24_Dis_mean = fast_cumsum(M24_Dis, MC, 400)
M24_Ord_mean = fast_cumsum(M24_Ord, MC, 400)

#Plotting Energy
fig, ax = plt.subplots(2, 2, sharex=True)

ax[0, 0].plot(MC[: int(1e4)], E24_Dis_mean[: int(1e4)])
ax[0, 0].set_title(r"$K_BT/J = 2.4$, Disordered")
ax[0, 0].set_ylabel(r"$Energy [J]$")

ax[1, 0].plot(MC[: int(1e4)], E24_Ord_mean[: int(1e4)])
ax[1, 0].set_title(r"$K_BT/J = 2.4$, Ordered")
ax[1, 0].set_xlabel("# Monte Carlo cycles")
ax[1, 0].set_ylabel(r"$Energy [J]$")

ax[0, 1].plot(MC[: int(1e4)], E1_Dis_mean[: int(1e4)])
ax[0, 1].set_title(r"$K_BT/J = 1$, Disordered")
ax[0, 1].set_ylabel(r"$Energy [J]$")

ax[1, 1].plot(MC[: int(1e4)], E1_Ord_mean[: int(1e4)])
ax[1, 1].set_title(r"$K_BT/J = 1$, Ordered")
ax[1, 1].set_ylabel(r"$Energy [J]$")
ax[1, 1].set_xlabel("# Monte Carlo cycles")

fig.tight_layout(w_pad=1)
fig.set_size_inches(7.1014, 9.0971 / 2)
plt.savefig("../doc/Figures/E_MC1e7.pdf")
plt.figure()

#Plotting number of flips
fig, ax = plt.subplots(2, 2, sharex=True)

ax[0, 0].plot(MC, flip24_Dis)
ax[0, 0].set_title(r"$K_BT/J = 2.4$, Disordered")
ax[0, 0].set_ylabel("# Flips")


ax[1, 0].plot(MC, flip24_Ord)
ax[1, 0].set_title(r"$K_BT/J = 2.4$, Ordered")
ax[1, 0].set_xlabel("# Monte Carlo cycles")
ax[1, 0].set_ylabel("# Flips")

ax[0, 1].plot(MC, flip1_Dis)
ax[0, 1].set_title(r"$K_BT/J = 1$, Disordered")
ax[0, 1].set_ylabel("# Flips")

ax[1, 1].plot(MC, flip1_Ord)
ax[1, 1].set_title(r"$K_BT/J = 1$, Ordered")
ax[1, 1].set_xlabel("# Monte Carlo cycles")
ax[1, 1].set_ylabel("# Flips")
fig.tight_layout(w_pad=1)
fig.set_size_inches(7.1014, 9.0971 / 2)
plt.savefig("../doc/Figures/flip_MC1e7.pdf")
plt.figure()

#Plotting magnetization
fig, ax = plt.subplots(2, 2, sharex=True)

ax[0, 0].plot(MC[: int(1e4)], M24_Dis_mean[: int(1e4)])
ax[0, 0].set_title(r"$K_BT/J = 2.4$, Disordered")
ax[0, 0].set_ylabel(r"$\vert M \vert$")


ax[1, 0].plot(MC[: int(1e4)], M24_Ord_mean[: int(1e4)])
ax[1, 0].set_title(r"$K_BT/J = 2.4$, Ordered")
ax[1, 0].set_xlabel("# Monte Carlo cycles")
ax[1, 0].set_ylabel(r"$\vert M \vert$")

ax[0, 1].plot(MC[: int(1e4)], M1_Dis_mean[: int(1e4)])
ax[0, 1].set_title(r"$K_BT/J = 1$, Disordered")
ax[0, 1].set_ylabel(r"$\vert M \vert$")

ax[1, 1].plot(MC[: int(1e4)], M1_Ord_mean[: int(1e4)])
ax[1, 1].set_title(r"$K_BT/J = 1$, Ordered")
ax[1, 1].set_xlabel("# Monte Carlo cycles")
ax[1, 1].set_ylabel(r"$\vert M \vert$")
fig.tight_layout(w_pad=1)
fig.set_size_inches(7.1014, 9.0971 / 2)
plt.savefig("../doc/Figures/M_MC1e7.pdf")
plt.figure()

bins24 = np.arange(
    np.min(E24_Dis[int(5e3) :] / 400), np.max(E24_Dis[int(5e3) :] / 400), 4 / 400
)
bins1 = np.arange(
    np.min(E1_Dis[int(5e3) :] / 400), np.max(E1_Dis[int(5e3) :] / 400), 4 / 400
)

fig, ax = plt.subplots(2, 1)
ax[0].hist(E24_Dis[int(5e3) :] / 400, bins=bins24, density=True)
ax[1].hist(E1_Dis[int(5e3) :] / 400, bins=bins1, density=True)
ax[0].set_title(r"$K_BT/J = 2.4$")
ax[1].set_title(r"$K_BT/J = 1$")
fig.set_size_inches(3.35289, 9.0971 / 2)
ax[1].set_xlabel(r"$E / N^2$ [J]")
ax[0].set_ylabel(r"% of occurences")
ax[1].set_ylabel(r"% of ocurrences")
fig.tight_layout(w_pad=1)
plt.savefig("../doc/Figures/histogram.pdf")

#Calculating variance
Var24 = np.var(E24_Dis[int(5e3):]/400) #/ MC[-1]
Var1 = np.var(E1_Dis[int(5e3):]/400)
print("Var(E24_Dis) = {0}, Error estimate = {1}".format(Var24, Var24/MC[-1]))
print("Var(E1_Dis) = {0}, Error estimate = {1}".format(Var1, Var1/MC[-1]))
