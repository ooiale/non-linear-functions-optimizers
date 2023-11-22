import numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
from autograd import grad
import autograd.numpy as np
from numpy.linalg import solve
import functions as f



def cp1_dfp (function, a, b, y, s, p, e, m, x0 ,h0, method = 'cp1'):
    n = len(x0)
    k = 0
    xk = x0
    dk = np.zeros(len(x0))
    hk = h0
    gradient = egrad(function)
    n = len(x0)
    while ((np.linalg.norm(gradient(xk, n))) >= e) and (k < m):
        gradXk = gradient(xk, n)
        dk = -(hk @ gradXk)

        
        if (np.dot(gradXk,dk) > -y * np.linalg.norm(gradXk) * np.linalg.norm(dk)):
            hk = np.eye(n)
            dk = -gradXk

        if np.linalg.norm(dk) < b * np.linalg.norm(gradXk):
            dk = b * np.linalg.norm(gradXk) / np.linalg.norm(dk) * dk

        tk = 1
        while (function(xk + tk*dk, n) > function(xk, n) + a*tk*np.dot(gradXk, dk)):
            tk = s*tk
        xk =xk + tk*dk
        
        xkk = xk + tk*dk
        sk = xkk - xk
        yk = gradient(xkk, n) - gradient(xk, n)
        zk = hk @ yk
        wk = sk - zk

        if method == 'cp1':
            if np.dot(wk,yk) > 0:
                hk = hk + (np.outer(wk,wk) / np.dot(wk,yk))
            k = k + 1

        elif method == 'dfp':
            if np.dot(sk,yk) > 0:
                hk = hk + (np.outer(sk,sk) / np.dot(sk,yk)) - (np.outer(zk,zk) / np.dot(zk,yk))
            k = k + 1

    return xk,k


#def function(x, n):
#    fun = 0
#    for i in range(n):
#        fun = fun + x[i]**2 - 10 * np.cos(np.pi * 2 * x[i])
#    return fun

if __name__ == "__main__":

    n = 2
    ponto = np.array ( [0.0]*n )
    alpha = 10e-4
    beta = 10e-3
    gamma = 10e-6
    sigma = 0.5
    ro = 10e-3
    m = 100
    epslon = 10e-6
    h0 = np.eye(n)
    metodos = ['dfp', 'cp1']
    functions = [f.Quadratica, f.Rosenbrook, f.StyblinskyTang, f.Rastrigin]
    for metodo in metodos:
        for fun in functions:
            x_otimo , k =  cp1_dfp(fun, alpha, beta, gamma, sigma, ro, epslon, m, ponto, h0, method=metodo)
            print(f'Metodo {metodo} na funcao {fun.__name__}:')
            print(f'Ponto inicial: {ponto}')
            print (f'Minimizador local: {np.around(x_otimo, decimals=3)} em {k} iteracoes')
            print(f'f(x) = {np.around(fun(x_otimo, n), decimals = 3)}')
            print()


'''
x_value = np.array([2.0, 2.0])  # note inputs have to be floats
gradient = egrad(function)
H_f = jacobian(gradient)  # returns a function
print(H_f(x_value))

gradient_at_x = gradient(x_value)
print(gradient_at_x)
'''