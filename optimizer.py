from scipy.optimize import minimize
import numpy as np


class optimizer:
    def __init__(self, Y_p, alpha, dim_reduction=True, method='SQP'):
        self.Y_p = Y_p
        self.alpha = alpha
        self.dim_reduction = dim_reduction
        self.method = method

    def objective(self,Omega_k):
            Omega_k = Omega_k.reshape(self.Omega_k0.shape)
            res = np.sum(np.sqrt(np.sum(Omega_k ** 2, axis=1)))
            return res
    def constraint(self,Omega_k):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        return (self.Y_p @ Omega_k - self.Uk @ self.Lambda_k).flatten()
    
    def SQP(self,Omega_k0):
        con = {'type': 'eq', 'fun': self.constraint}
        return minimize(self.objective, Omega_k0.flatten(), constraints=con, method='SLSQP')
    
    def GD_lagrange_multi(self,Omega_k0,mu=1e-3):
        def grad_omega_k(Omega_k,lagrange_multi_k):
            Omega_k = Omega_k.reshape(self.Omega_k0.shape)
            grad = Omega_k / np.sqrt(np.sum(Omega_k ** 2, axis=1))[:,None]
            grad += self.Y_p.T @ lagrange_multi_k.reshape(self.Y_p.shape[0],self.Y_p.shape[0])
            return grad
        def grad_lagrange_multi_k(Omega_k):
            return self.constraint(Omega_k)
        
        lagrange_multi_k = np.zeros((self.Y_p.shape[0]**2,1))
        Omega_k = np.random.randn(Omega_k0.shape[0],Omega_k0.shape[1])
        for iter in range(100000):
            grad_omega = grad_omega_k(Omega_k,lagrange_multi_k)
            grad_lagrange_multi = grad_lagrange_multi_k(Omega_k)

            Omega_k -= mu * grad_omega
            lagrange_multi_k += mu * grad_lagrange_multi.reshape(-1,1)

        return Omega_k

    def unmix_and_smooth(self,Bk,Omega_k_opt,D_prior):
        Dk = Omega_k_opt @ np.linalg.pinv(self.Uk @ self.Lambda_k)
        if D_prior is not None:
            Dk = self.alpha * Dk + (1 - self.alpha) * D_prior
        Sk = Dk @ Bk
        return Sk,Dk

    def optimize(self,Bk,D_prior):
        #Dim Reduction
        self.Uk, Lk, Vkt = np.linalg.svd(Bk)
        self.Lambda_k = np.diag(Lk)  # diagonal matrix of singular values
        self.Omega_k0 = np.zeros((self.Y_p.shape[1], self.Lambda_k.shape[1]))# init Omega_k

        if self.method == 'SQP':
            res = self.SQP(self.Omega_k0)
            Omega_k_opt = res.x.reshape(self.Omega_k0.shape)

        if self.method == 'GD_lagrange_multi':
            Omega_k_opt = self.GD_lagrange_multi(self.Omega_k0)


        Sk, Dk = self.unmix_and_smooth(Bk,Omega_k_opt,D_prior)
        return Sk, Dk