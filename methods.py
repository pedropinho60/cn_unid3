import numpy as np
from typing import Tuple, Callable

def _linear_regression_base(x_points: np.ndarray, y_points: np.ndarray) -> Tuple[float, float]:
    n = x_points.size
    if n == 0:
        raise ValueError("x_points array cannot be empty")
    if y_points.size != n:
        raise ValueError("x_points and y_points must have the same size")

    x_sum = np.sum(x_points)
    y_sum = np.sum(y_points)
    xy_sum = np.sum(x_points * y_points)
    x2_sum = np.sum(x_points**2)

    denominator = n * x2_sum - x_sum**2
    if denominator == 0:
        raise ValueError("Cannot compute linear regression - denominator is zero (possibly all x values are the same)")

    a = (n * xy_sum - x_sum * y_sum) / denominator
    b = (x_sum * xy_sum - y_sum * x2_sum) / -denominator

    return a, b

def linear_regression(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
    a,b = _linear_regression_base(x_points, y_points)
    return lambda x: a*x + b

def logarithmic_regression(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
    ln_x_points = np.log(x_points)
    a,b = _linear_regression_base(ln_x_points, y_points)
    return lambda x: a*np.log(x) + b

def exponential_regression(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
    ln_y_points = np.log(y_points)
    a,ln_b = _linear_regression_base(x_points, ln_y_points)
    b = np.exp(ln_b)
    return lambda x: b * np.exp(a*x)

def power_regression(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
    ln_x_points = np.log(x_points)
    ln_y_points = np.log(y_points)
    a,ln_b = _linear_regression_base(ln_x_points, ln_y_points)
    b = np.exp(ln_b)
    return lambda x: b * x**a

def geometric_regression(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
    ln_y_points = np.log(y_points)
    ln_a,ln_b = _linear_regression_base(x_points, ln_y_points)
    a = np.exp(ln_a)
    b = np.exp(ln_b)
    return lambda x: b * a**x

def lu_decomposition(A):
    A = A.astype(float)
    n = A.shape[0]

    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n):

        pivot_row = np.argmax(np.abs(U[k:, k])) + k

        if U[pivot_row, k] == 0:
            raise ValueError("Matrix is singular (no unique solution).")


        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        for i in range(k + 1, n):
            fator = U[i, k] / U[k, k]
            L[i, k] = fator
            U[i, :] -= fator * U[k, :]

    return P, L, U

def lu_method(A, b):
    P, L, U = lu_decomposition(A)
    Pb = P @ b

    n = A.shape[0]


    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]


    x = np.zeros_like(b, dtype=float)
    for i in reversed(range(n)):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x

def quadratic_regression(x_points: np.ndarray, y_points: np.ndarray) -> Callable[[float], float]:
    n = x_points.size
    if n == 0:
        raise ValueError("x_points array cannot be empty")
    if y_points.size != n:
        raise ValueError("x_points and y_points must have the same size")

    sum_x4 = np.sum(x_points**4)
    sum_x3 = np.sum(x_points**3)
    sum_x2 = np.sum(x_points**2)
    sum_x = np.sum(x_points)
    sum_x2y = np.sum((x_points**2) * y_points)
    sum_xy = np.sum(x_points * y_points)
    sum_y = np.sum(y_points)

    A : np.ndarray= np.array([
        [sum_x4, sum_x3, sum_x2],
        [sum_x3, sum_x2, sum_x],
        [sum_x2,  sum_x, n]
    ])
    B : np.ndarray= np.array([
        sum_x2y,
        sum_xy,
        sum_y
    ])

    a, b, c = lu_method(A, B)

    return lambda x: a * (x**2) + b * x + c
