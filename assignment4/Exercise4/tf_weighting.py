import math
import pdb
import numpy as np

term_freq_mat = [
[1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00],
[1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00],
[0.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
[0.00, 1.00, 1.00, 2.00, 0.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
[0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 0.00],
[0.00, 1.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00]
]
T_term_freq_mat = np.zeros((len(term_freq_mat), len(term_freq_mat[0])))
n = len(term_freq_mat[0])

for row in range(len(term_freq_mat)):
    for column in range(len(term_freq_mat[row])):
        f_ij = term_freq_mat[row][column]
        X_f_ij = 0
        for freq in term_freq_mat[row]:
            if freq > 0.00:
                X_f_ij = X_f_ij + 1.0
        term = n / X_f_ij
        T_term_freq_mat[row][column] = f_ij * np.log( term)

from pandas import *
x = T_term_freq_mat
print(DataFrame(x))
