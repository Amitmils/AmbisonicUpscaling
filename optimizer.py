from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class optimizer:
    def __init__(
        self,
        Y_p,
        alpha,
        dim_reduction=True,
        constraint_tol=0,
        method="GD_lagrange_multi",
    ):
        self.Y_p = Y_p
        self.num_SH_coeff = Y_p.shape[0]
        self.num_grid_points = Y_p.shape[1]
        self.alpha = alpha
        self.dim_reduction = dim_reduction
        self.method = method
        self.constraint_tol = constraint_tol

    def objective(self, Omega_k):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        res = np.sum(np.sqrt(np.sum(Omega_k**2, axis=1)))
        return res

    def constraint(self, Omega_k, Y_p=None):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        if Y_p is None:
            Y_p = self.Y_p
        constraint_res = (
            np.matmul(Y_p, Omega_k) - np.matmul(self.Uk, self.Lambda_k)
        ).reshape(self.num_windows, self.num_bins, -1)
        return constraint_res

    def SQP(self, Omega_k0):
        con = {"type": "eq", "fun": self.constraint}
        return minimize(
            self.objective, Omega_k0.flatten(), constraints=con, method="SLSQP"
        )

    def GD_lagrange_multi(self, Omega_k0, mu=1e-3, ro=1e-3):
        def grad_omega_k(Omega_k, lagrange_multi_k):
            Omega_k = Omega_k.reshape(self.Omega_k0.shape)
            grad = Omega_k / np.sqrt(np.sum(Omega_k**2, axis=-1))[..., None]
            reshaped_lagrange_multi_k = lagrange_multi_k.reshape(
                self.num_windows,
                self.num_bins,
                self.num_SH_coeff,
                self.num_SH_coeff,
            )
            transposed_Yp = np.transpose(reduced_Yp, (0, 1, 3, 2))
            broadcasted_transposed_Yp = np.broadcast_to(
                transposed_Yp,
                (
                    self.num_windows,
                    self.num_bins,
                    transposed_Yp.shape[-2],
                    transposed_Yp.shape[-1],
                ),
            )
            grad += np.matmul(
                broadcasted_transposed_Yp,
                reshaped_lagrange_multi_k,
            )
            return grad

        def grad_lagrange_multi_k(Omega_k):
            return self.constraint(Omega_k, reduced_Yp)

        # Omega_k0 is (window,bin,*,**)
        lagrange_multi_k = np.zeros(
            (self.num_windows, self.num_bins, self.num_SH_coeff**2, 1)
        )
        Omega_k = np.random.randn(*Omega_k0.shape)
        for iter in tqdm(range(int(1e5))):
            non_zero_indices_in_grid = np.nonzero(self.mask[0,0,:])[0]
            reduced_Yp = self.Y_p[:,:,:,non_zero_indices_in_grid].reshape(
                self.num_windows, self.num_bins, self.num_SH_coeff, -1
            )  # TODO Currently assumes mask doesnt change over time
            grad_omega = grad_omega_k(Omega_k, lagrange_multi_k)
            grad_lagrange_multi = grad_lagrange_multi_k(Omega_k)
            Omega_k -= mu * grad_omega
            if self.constraint_tol == 0:
                # equlity constraint
                lagrange_multi_k += ro * grad_lagrange_multi[..., None]
            else:
                # inequlity constraint
                for i in range(len(grad_lagrange_multi)):
                    if grad_lagrange_multi[i] > self.constraint_tol:
                        lagrange_multi_k[i] += ro * (
                            grad_lagrange_multi[i] - self.constraint_tol
                        )
                    elif grad_lagrange_multi[i] < -self.constraint_tol:
                        lagrange_multi_k[i] += ro * (
                            grad_lagrange_multi[i] + self.constraint_tol
                        )
                    else:
                        lagrange_multi_k[i] = max(0, lagrange_multi_k[i])
        self.constraint_loss = self.constraint(Omega_k, reduced_Yp)  # not really a loss
        self.objective_loss = self.objective(Omega_k)
        return Omega_k

    def unmix_and_smooth(self, Bk, Omega_k_opt, D_prior):
        Dk = np.matmul(
            Omega_k_opt,
            torch.linalg.pinv(torch.tensor(np.matmul(self.Uk, self.Lambda_k))).numpy(),
        )
        if D_prior is not None:
            Dk = self.alpha * Dk + (1 - self.alpha) * D_prior
        Sk = np.matmul(Dk, Bk)
        return Sk, Dk

    def SLS(self):  # Sequential Least Squares
        # TODO doesnt support batches
        Omega_opt = np.linalg.pinv(self.Y_p[:, self.mask]) @ self.Uk @ self.Lambda_k
        return Omega_opt

    def optimize(self, Bk, mask=None, D_prior=None):
        if isinstance(Bk, tuple):
            # cheap hack for MP
            tmp = Bk
            Bk = tmp[0]
            mask = tmp[1]
            D_prior = tmp[2]
        self.num_windows, self.num_bins = Bk.shape[:2]

        # Dim Reduction
        if mask is None:
            mask = np.ones(
                (self.num_windows, self.num_bins, self.Y_p.shape[1]), dtype=bool
            )
        else:
            if mask.shape[0] == 1 and mask.shape[1] == 1:
                mask = np.broadcast_to(mask, (self.num_windows, self.num_bins, mask.shape[-1]))
        
        num_non_zero_indices_in_grid = np.count_nonzero(mask[0,0,:]) #ASSUMES MASK DOESNT CHANGE OVER TIME
        self.Y_p = np.broadcast_to(self.Y_p, (self.num_windows, self.num_bins,*self.Y_p.shape))
        self.mask = mask
        self.Uk, Lk, Vkt = torch.linalg.svd(torch.tensor(Bk), full_matrices=False)
        self.Uk = self.Uk.numpy()
        self.Lambda_k = torch.diag_embed(Lk).numpy()
        self.Omega_k0 = np.zeros(
            (
                self.num_windows,
                self.num_bins,
                num_non_zero_indices_in_grid, 
                self.Lambda_k.shape[-1],
            )
        )  # init Omega_k

        if self.method == "SQP":
            res = self.SQP(self.Omega_k0)
            Omega_k_opt = res.x.reshape(self.Omega_k0.shape)

        if self.method == "GD_lagrange_multi":
            Omega_k_opt = self.GD_lagrange_multi(self.Omega_k0)

        if self.method == "SLS":
            Omega_k_opt = self.SLS()

        Sk = np.zeros(
            (self.num_windows, self.num_bins, self.num_grid_points, Bk.shape[-1])
        )  # (num windows,num bins,SH Coeff, Window length)
        #Sk[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2], :]
        Sk_temp, Dk = (
            self.unmix_and_smooth(Bk, Omega_k_opt, D_prior)
        )
        non_zero_indices_in_grid = np.nonzero(mask[0,0,:])
        for w in range(self.num_windows):
            for b in range(self.num_bins):
                Sk[w, b, non_zero_indices_in_grid, :] = Sk_temp[w, b, :, :]
        return Sk, Dk
