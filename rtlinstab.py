import sympy as sp
import numpy as np
from joblib import Parallel, delayed


def get_equation(k_value, n):
    # 未知数
    # 定数パラメータ
    a, b = 1.6, 1.604  # rho_2, rho_1
    c, d = 1.1, 1.1  # nu_2, nu_1
    g, T = 1, 0.98  # 重力と界面張力
    A = (b - a) / (b + a)  # Atwood数
    k = k_value

    # よく出てくる平方根
    sqrt_c = sp.sqrt(k**2 + n / c)
    sqrt_d = sp.sqrt(k**2 + n / d)

    # 方程式
    equation = (
        -((k / n**2) * (T * k**2 / (a + b) - A * g) + 1)
        * ((b * sqrt_c + a * sqrt_d) / (a + b) - k)
        + 4
        * (a * c - b * d)
        * k**2
        / ((a + b) * n)
        * ((b * sqrt_c - a * sqrt_d) / (a + b) - A * k)
        - 4 * a * b * k / (a + b) ** 2
        + 4
        * (a * c - b * d) ** 2
        * k**3
        / ((a + b) ** 2 * n**2)
        * (sqrt_c - k)
        * (sqrt_d - k)
    )
    return equation


def mysolve(k_value):
    n = sp.Symbol("n")
    equation = get_equation(k_value, n)
    solutions = sp.solve(equation, n)
    return solutions[0]


def mynpsolve(k_value, guess):
    n = sp.Symbol("n")
    equation = get_equation(k_value, n)
    # 数値解を複素数で
    solutions = sp.nsolve(equation, n, guess)
    return solutions


def main():
    kstart = 0.005
    kend = 0.14
    N = 10
    k_values = np.linspace(kstart, kend, N + 1)
    results = Parallel(n_jobs=-1)(delayed(mysolve)(k) for k in k_values)

    # 結果を表示
    for k, sol in zip(k_values, results):
        print(f"k={k:.5f}, n={sol}")


if __name__ == "__main__":
    main()
