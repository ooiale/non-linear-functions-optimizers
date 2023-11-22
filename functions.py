import autograd.numpy as np


def Rastrigin(x, n):
    fun = 0
    for i in range(n):
        fun = fun + x[i]**2 - 10 * np.cos(np.pi * 2 * x[i])
    return fun

def StyblinskyTang(x, n):
    fun = 0
    for i in range(n):
        fun = fun + x[i]**4 - 16 * x[i]**2 + 5*x[i]
    return fun

def Rosenbrook(x, n):
    if n % 2 != 0:
        raise ValueError("N needs to be even")

    fun = 0
    for i in range(1, n // 2 + 1):
        fun = fun + 10 * (x[2 * i - 1] - x[2 * i - 2] ** 2) ** 2 + (x[2 * i - 2] - 1) ** 2
    return fun

def Quadratica(x, n):
    fun = 0
    for i in range(n):
        fun = fun + (i + 1)* x[i]**2
    return fun

if __name__ == "__main__":
    print('oi')
