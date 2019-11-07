import numpy as np
import matplotlib.pyplot as plt
import numba as nb

def read_file(filename, skip=2):
    return np.loadtxt(filename, skiprows=skip), filename

@nb.njit
def save_npy(file, filename):
    np.save(filename[:-3] + "npy", file)

save_npy(read_file_fast("P_T1_MC1e7_Disordered.txt"))
save_npy(read_file_fast("P_T1_MC1e7_Ordered.txt"))
save_npy(read_file_fast("P_T24_MC1e7_Disordered.txt"))
save_npy(read_file_fast("P_T24_MC1e7_Ordered.txt"))
