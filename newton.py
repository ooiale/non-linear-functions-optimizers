import numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np
from numpy.linalg import solve
import functions as f

def newton(function, alpha, b, y, s, p, e, m, x0, n):
    k = 0
    xk = x0
    dk = np.zeros(n)
    gradient = egrad(lambda x: function(x, n))
    n = len(x0)
    
    while ((np.linalg.norm(gradient(xk))) >= e) and (k < m):
        gradXk = gradient(xk)
        u = 0
        a = 0
        while a == 0:
            try:
                H_f = jacobian(gradient)  # returns a function
                augmented_hessian = H_f(xk) + u * np.eye(n)

                # Solve the linear system (H + uI) * dk = -gradient for dk
                dk = solve(augmented_hessian, -gradXk)

                if (np.dot(gradXk, dk) > -y * np.linalg.norm(gradXk) * np.linalg.norm(dk)):
                    u = max(2 * u, p)
                else:
                    a = 1
            except Exception:
                u = max(2 * u, p)

        if np.linalg.norm(dk) < b * np.linalg.norm(gradXk):
            dk = b * np.linalg.norm(gradXk) / np.linalg.norm(dk) * dk

        tk = 1
        while (function(xk + tk * dk, n) > function(xk, n) + alpha * tk * np.dot(gradXk, dk)):
            tk = s * tk
        xk = xk + tk * dk
        k = k + 1

    return xk, k

#def function(x, n):
#    fun = 0
#    for i in range(n):
#        fun = fun + x[i]**2 - 10 * np.cos(np.pi * 2 * x[i])
#    return fun

if __name__ == "__main__":
    dimension_n = 2 
    point = np.array([0.0] * dimension_n ) 
    alpha = 1e-4
    beta = 1e-3
    gamma = 1e-6
    sigma = 0.5
    ro = 1e-3
    m = 100
    epsilon = 1e-6

    functions = [f.Quadratica, f.Rosenbrook, f.StyblinskyTang, f.Rastrigin]
    for fun in functions:
        x_optimal, num_iterations = newton(fun, alpha, beta, gamma, sigma, ro, epsilon, m, point, dimension_n)
        print(f'Metodo de Newton na funcao {fun.__name__}:')
        print(f'Ponto inicial: {point}')
        print(f'minimizador local: {np.around(x_optimal, decimals=3)} em {num_iterations} iteracoes')
        print(f'f(x) = {np.around(fun(x_optimal, dimension_n), decimals = 3)}')
        print()