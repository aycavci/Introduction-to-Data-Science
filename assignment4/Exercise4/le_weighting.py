import numpy as np
import math
import pdb

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
w = np.sum(term_freq_mat, axis = 0)
n = len(term_freq_mat[0])

for row in range(len(term_freq_mat)):
    for column in range(len(term_freq_mat[row])):
        term = 0
        for freq in term_freq_mat[row]:
            p_ij = freq / sum(term_freq_mat[row])
            if p_ij > 0.0:
                term = term + (p_ij * math.log(p_ij))
            else:
                term = term + 0


        f_ij = term_freq_mat[row][column]
        #pdb.set_trace()
        #pdb.set_trace()
        T_term_freq_mat[row][column] = math.log(1 + f_ij) * (1 + (term / math.log(n)))

from pandas import *
x = T_term_freq_mat
print(DataFrame(x))
