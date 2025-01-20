from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


class optimizer:
    def __init__(self, Y_p, alpha, dim_reduction=True,constraint_tol=0, method='GD_lagrange_multi'):
        self.Y_p = Y_p
        self.alpha = alpha
        self.dim_reduction = dim_reduction
        self.method = method
        self.constraint_tol = constraint_tol

    def objective(self,Omega_k):
            Omega_k = Omega_k.reshape(self.Omega_k0.shape)
            res = np.sum(np.sqrt(np.sum(Omega_k ** 2, axis=1)))
            return res
    def constraint(self,Omega_k):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        constraint_res = (self.Y_p[:,self.mask] @ Omega_k - self.Uk @ self.Lambda_k).flatten()
        return constraint_res
    
    def SQP(self,Omega_k0):
        con = {'type': 'eq', 'fun': self.constraint}
        return minimize(self.objective, Omega_k0.flatten(), constraints=con, method='SLSQP')
    
    def GD_lagrange_multi(self,Omega_k0,mu=1e-3,ro = 1e-3):
        def grad_omega_k(Omega_k,lagrange_multi_k):
            Omega_k = Omega_k.reshape(self.Omega_k0.shape)
            grad = Omega_k / np.sqrt(np.sum(Omega_k ** 2, axis=1))[:,None]
            grad += self.Y_p[:,self.mask].T @ lagrange_multi_k.reshape(self.Y_p.shape[0],self.Y_p.shape[0])
            return grad
        def grad_lagrange_multi_k(Omega_k):
            return self.constraint(Omega_k)
        
        lagrange_multi_k = np.zeros((self.Y_p.shape[0]**2,1))
        Omega_k = np.random.randn(Omega_k0.shape[0],Omega_k0.shape[1])
        self.loss = list()
        for iter in range(int(30000)):
            grad_omega = grad_omega_k(Omega_k,lagrange_multi_k)
            grad_lagrange_multi = grad_lagrange_multi_k(Omega_k)
            Omega_k -= mu * grad_omega
            if self.constraint_tol == 0:
                #equlity constraint
                lagrange_multi_k += ro * grad_lagrange_multi.reshape(-1,1)
            else:
                #inequlity constraint
                for i in range(len(grad_lagrange_multi)):
                    if grad_lagrange_multi[i] > self.constraint_tol:
                        lagrange_multi_k[i] += ro * (grad_lagrange_multi[i] - self.constraint_tol)
                    elif grad_lagrange_multi[i] < -self.constraint_tol:
                        lagrange_multi_k[i] += ro * (grad_lagrange_multi[i] + self.constraint_tol)
                    else:
                        lagrange_multi_k[i] = max(0, lagrange_multi_k[i])
            self.loss.append(10*np.log10(np.linalg.norm(self.constraint(Omega_k),ord=2)**2))
        print(iter)
        self.constraint_loss = self.constraint(Omega_k) #not really a loss
        self.objective_loss = self.objective(Omega_k)
        return Omega_k

    def unmix_and_smooth(self,Bk,Omega_k_opt,D_prior):
        Dk = Omega_k_opt @ np.linalg.pinv(self.Uk @ self.Lambda_k)
        if D_prior is not None:
            Dk = self.alpha * Dk + (1 - self.alpha) * D_prior
        Sk = Dk @ Bk
        return Sk,Dk

    def SLS(self): #Sequential Least Squares
        Omega_opt = np.linalg.pinv(self.Y_p[:,self.mask]) @ self.Uk @ self.Lambda_k
        return Omega_opt

    def optimize(self,Bk,mask=None,D_prior=None):
        if isinstance(Bk,tuple):
            #cheap hack for MP
            tmp = Bk
            Bk = tmp[0]
            mask = tmp[1]
            D_prior = tmp[2]

        #Dim Reduction
        if mask is None:
            mask = np.ones(self.Y_p.shape[1],dtype=bool)
        self.mask = mask
        self.Uk, Lk, Vkt = np.linalg.svd(Bk)
        self.Lambda_k = np.diag(Lk)  # diagonal matrix of singular values
        self.Omega_k0 = np.zeros((np.count_nonzero(mask), self.Lambda_k.shape[1]))# init Omega_k

        if self.method == 'SQP':
            res = self.SQP(self.Omega_k0)
            Omega_k_opt = res.x.reshape(self.Omega_k0.shape)

        if self.method == 'GD_lagrange_multi':
            Omega_k_opt = self.GD_lagrange_multi(self.Omega_k0)
        
        if self.method == 'SLS':
            Omega_k_opt = self.SLS()

        Sk = np.zeros((self.Y_p.shape[1],Bk.shape[1]))
        Sk[self.mask,:], Dk = self.unmix_and_smooth(Bk,Omega_k_opt,D_prior)
        return Sk, Dk , self.loss