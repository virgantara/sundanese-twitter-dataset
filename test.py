

import numpy as np
from numpy import linalg as la
from sympy import Matrix

A = np.array([
    [1, -2],
    [1, 4]
])

m = Matrix(A)
# eigvals, eigvecs = la.eig(A)
#
# eigvecs = np.real_if_close(eigvecs, tol=1)
#
# print(eigvecs * 10**.5)

vecs = m.eigenvects()
print(vecs)