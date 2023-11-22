import numpy as np
from autograd import elementwise_grad as egrad
import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functions as f


#def function(x, n):
#    fun = 0
#    for i in range(n):
#        fun = fun + x[i]**2 - 10 * np.cos(np.pi * 2 * x[i])
#    return fun


def grad_with_plot(function, a, s, e, m, x0, n):
    k = 0
    xk = x0
    dk = np.zeros(n)
    gradient = egrad(lambda x: function(x, n))

    
    if n == 2:
        x = np.linspace(-4, 4, 400)
        y = np.linspace(-4, 4, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function([X[i, j], Y[i, j]], n)

        # plotar a funcao
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

    while ((np.linalg.norm(gradient(xk))) >= e) and (k < m):
        gradXk = gradient(xk)
        dk = -gradXk
        tk = 1
        while (function(xk + tk * dk, n) > function(xk, n) + a * tk * np.dot(gradXk, dk)):
            tk = s * tk
        xk = xk + tk * dk
        k = k + 1
        # plotar os pontos se n = 2
        if n == 2:
            #ax.scatter(xk[0], xk[1], function(xk, n), color='red')
            pass

    if n == 2:
        #plt.show()
        pass
    return xk, k

def gradiente():
    alpha = 1e-6
    sigma = 0.5
    m = 100
    epsilon = 1e-6
    dimension_n = 2  #numero de dimensao
    point = np.array([0.0, 0.0])  # valor do ponto inicial
    functions = [f.Quadratica, f.Rosenbrook, f.StyblinskyTang, f.Rastrigin]
    for fun in functions:
        x_optimal, num_iterations = grad_with_plot(fun, alpha, sigma, epsilon, m, point, dimension_n)
        print(f'Metodo do gradiente na funcao {fun.__name__}:')
        print(f'Ponto inicial: {point}')
        print(f'minimizador local: {np.around(x_optimal, decimals=3)} em {num_iterations} iteracoes')
        print(f'f(x) = {np.around(fun(x_optimal, dimension_n),decimals = 3)}\n')


if __name__ == "__main__":
    gradiente()
