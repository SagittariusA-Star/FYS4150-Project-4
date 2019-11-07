import numpy as np
import matplotlib.pyplot as plt
import numba as nb

def read_file(filename, skip=2):
    return np.loadtxt(filename, skiprows=skip), filename

def save_npy(file, filename):
    np.save(filename[:-3] + "npy", file)

P_T1_MC1e7_Disordered = read_file("P_T1_MC1e7_Disordered.txt")
P_T1_MC1e7_Ordered = read_file("P_T1_MC1e7_Ordered.txt")
P_T24_MC1e7_Disordered = read_file("P_T24_MC1e7_Disordered.txt")
P_T24_MC1e7_Ordered = read_file("P_T24_MC1e7_Ordered.txt")

save_npy(P_T1_MC1e7_Disordered[0], P_T1_MC1e7_Disordered[1])
save_npy(P_T1_MC1e7_Ordered[0], P_T1_MC1e7_Ordered[1])
save_npy(P_T24_MC1e7_Disordered[0], P_T24_MC1e7_Disordered[1])
save_npy(P_T24_MC1e7_Ordered[0], P_T24_MC1e7_Ordered[1])
