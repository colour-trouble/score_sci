import numpy as np

"""
yaping: decide a whether a matrix is full rank in python-numpy
"""


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def is_full_rank(a):
    a = a.cpu().numpy()
    return min(a.shape[0], a.shape[1]) == np.linalg.matrix_rank(a)


if __name__ == "__main__":
    # yaping: assume the shape of the mask is (256, 256)
    a = np.random.rand(256, 256)
    print(is_full_rank(a))
