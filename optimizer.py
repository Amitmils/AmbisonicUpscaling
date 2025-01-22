from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
import utils


class optimizer(nn.Module):
    def __init__(
        self,
        Y_p,
        alpha,
        dim_reduction=True,
        constraint_tol=0,
        save_loss=False,
        method="GD_lagrange_multi",
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.save_loss = save_loss
        self.Y_p = Y_p.to(self.device)
        self.num_SH_coeff = Y_p.shape[0]
        self.num_grid_points = Y_p.shape[1]
        self.alpha = alpha
        self.dim_reduction = dim_reduction
        self.method = method
        self.constraint_tol = constraint_tol
        self.reconstruction_loss = torch.tensor([]).to(self.device)

    def objective(self, Omega_k):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        res = torch.sum(torch.sqrt(torch.sum(Omega_k**2, dim=1)))
        return res

    def constraint(self, Omega_k, Y_p=None):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        if Y_p is None:
            Y_p = self.Y_p
        constraint_res = (
            torch.matmul(Y_p, Omega_k) - torch.matmul(self.Uk, self.Lambda_k)
        ).reshape(self.num_windows, self.num_bins, -1)
        return constraint_res

    def SQP(self, Omega_k0):
        con = {"type": "eq", "fun": self.constraint}
        return minimize(
            self.objective, Omega_k0.flatten(), constraints=con, method="SLSQP"
        )

    def GD_Deep(self, num_iterations: int):
        def init_para():
            self.learned_mu = nn.Parameter(
                torch.full(
                    (num_iterations, 1, self.num_bins, 1, 1), 1e-3, device=self.device
                ),
                requires_grad=True,
            )
            self.learned_lambda = nn.Parameter(
                torch.full(
                    (num_iterations, 1, self.num_bins, 1, 1),
                    self.alpha,
                    device=self.device,
                ),
                requires_grad=True,
            )

        def loss(Omega_k: torch.tensor, iter: int):
            loss = self.constraint(Omega_k) + self.learned_lambda[
                iter
            ] * self.objective(Omega_k)

        def loss_grad(Omega_k: torch.tensor, iter: int):
            regularization_grad = (
                Omega_k / torch.sqrt(torch.sum(Omega_k**2, dim=-1))[..., None]
            )

            reconstruction_grad = 2 * torch.matmul(
                reduced_Yp.t(),
                (
                    torch.matmul(reduced_Yp, Omega_k)
                    - torch.matmul(self.Uk, self.Lambda_k)
                ),
            )
            return reconstruction_grad + self.learned_lambda[iter] * regularization_grad

        init_para()
        non_zero_indices_in_grid = torch.nonzero(self.mask[0, 0, :]).to(self.device)
        reduced_Yp = self.Y_p[:, non_zero_indices_in_grid].reshape(
            self.num_SH_coeff, -1
        )  # TODO Currently assumes mask doesnt change over time

        Omega_k = torch.randn(*self.Omega_k0.shape).to(self.device)
        for iter in tqdm(range(int(num_iterations))):
            Omega_k -= self.learned_mu[iter] * loss_grad(Omega_k, iter)

            if self.save_loss:
                reconstruction_loss_per_window_freq_bin = 10 * torch.log10(
                    torch.norm(self.constraint(Omega_k), p=2, dim=-1) ** 2
                ).unsqueeze(0)
                self.reconstruction_loss = torch.cat(
                    (self.reconstruction_loss, reconstruction_loss_per_window_freq_bin),
                    dim=0,
                )

        return Omega_k

    def GD_lagrange_multi(self, iter=1e5, mu=1e-3, ro=1e-3):
        def grad_omega_k(Omega_k, lagrange_multi_k):
            Omega_k = Omega_k.reshape(self.Omega_k0.shape)
            grad = Omega_k / torch.sqrt(torch.sum(Omega_k**2, dim=-1))[..., None]
            reshaped_lagrange_multi_k = lagrange_multi_k.reshape(
                self.num_windows,
                self.num_bins,
                self.num_SH_coeff,
                self.num_SH_coeff,
            )
            grad += torch.matmul(
                reduced_Yp.t(),
                reshaped_lagrange_multi_k,
            )
            return grad

        def grad_lagrange_multi_k(Omega_k):
            return self.constraint(Omega_k, reduced_Yp)

        # Omega_k0 is (window,bin,*,**)
        lagrange_multi_k = torch.zeros(
            (self.num_windows, self.num_bins, self.num_SH_coeff**2, 1)
        ).to(self.device)
        Omega_k = torch.randn(*self.Omega_k0.shape).to(self.device)
        self.reconstruction_loss = self.reconstruction_loss.to(self.device)
        non_zero_indices_in_grid = torch.nonzero(self.mask[0, 0, :]).to(self.device)
        reduced_Yp = self.Y_p[:, non_zero_indices_in_grid].reshape(
            self.num_SH_coeff, -1
        )  # TODO Currently assumes mask doesnt change over time
        for iii in tqdm(range(int(iter))):
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
            if self.save_loss:
                reconstruction_loss_per_window_freq_bin = 10 * torch.log10(
                    torch.norm(self.constraint(Omega_k), p=2, dim=-1) ** 2
                ).unsqueeze(0)
                self.reconstruction_loss = torch.cat(
                    (self.reconstruction_loss, reconstruction_loss_per_window_freq_bin),
                    dim=0,
                )
        self.constraint_loss = self.constraint(Omega_k, reduced_Yp)  # not really a loss

        return Omega_k

    def unmix_and_smooth(self, Bk, Omega_k_opt, D_prior):
        Dk = torch.matmul(
            Omega_k_opt,
            torch.linalg.pinv(torch.matmul(self.Uk, self.Lambda_k)),
        ).cpu()
        if D_prior is not None:
            Dk = self.alpha * Dk + (1 - self.alpha) * D_prior
        Sk = torch.matmul(Dk, Bk)
        Sk_cpu = Sk.cpu()
        Dk_cpu = Dk.cpu()
        # del Sk
        # del Dk
        return Sk_cpu, Dk_cpu

    def SLS(self):  # Sequential Least Squares
        # TODO doesnt support batches
        Omega_opt = torch.linalg.pinv(self.Y_p[:, self.mask]) @ self.Uk @ self.Lambda_k
        return Omega_opt

    def optimize_v2(self, stft_anmt, iter=1e5,mu=1e-3, ro=1e-3):
        def grad_dict(s_t, lagrange_multi_k):
            grad = s_t / torch.sqrt(torch.sum(s_t**2, dim=-1))[..., None]
            grad += torch.matmul(
                self.Y_p.t(),
                lagrange_multi_k,
            )
            return grad

        def grad_lagrange_multi(s_t):
            constraint_res = torch.matmul(self.Y_p, s_t) - stft_anmt
            return constraint_res

        # stft_anmt is (window,bin,4)
        self.num_windows, self.num_channels,self.num_bins  = stft_anmt.shape
        lagrange_multi_t = torch.zeros(
            (self.num_windows, self.num_channels,self.num_bins),dtype=torch.complex32
        ).to(self.device)
        s_t = torch.randn(self.num_windows,self.num_grid_points, self.num_bins).to(
            self.device
        )
        self.reconstruction_loss = self.reconstruction_loss.to(self.device)
        for iii in tqdm(range(int(iter))):
            grad_s = grad_dict(s_t, lagrange_multi_t)
            grad_lagrange = grad_lagrange_multi(s_t)
            s_t -= mu * grad_s
            lagrange_multi_t += ro * grad_lagrange
        return s_t

    def optimize(self, Bk, itr=1e5, mask=None, D_prior=None, cheat=False):
        if isinstance(Bk, tuple):
            # cheap hack for MP
            tmp = Bk
            Bk = tmp[0]
            mask = tmp[1]
            D_prior = tmp[2]
        self.num_windows, self.num_bins = Bk.shape[:2]

        # Dim Reduction
        if mask is None:
            mask = torch.ones(
                (self.num_windows, self.num_bins, self.Y_p.shape[1]),
                dtype=bool,
                device=self.device,
            )
        else:
            if mask.shape[0] == 1 and mask.shape[1] == 1:
                mask = torch.broadcast_to(
                    mask, (self.num_windows, self.num_bins, mask.shape[-1])
                ).to(self.device)

        num_non_zero_indices_in_grid = torch.count_nonzero(
            mask[0, 0, :]
        )  # ASSUMES MASK DOESNT CHANGE OVER TIME
        self.mask = mask
        self.Uk, Lk, Vkt = torch.linalg.svd(Bk, full_matrices=False)
        self.Uk = self.Uk.to(self.device)
        self.Lambda_k = torch.diag_embed(Lk).to(self.device)
        self.Omega_k0 = torch.zeros(
            (
                self.num_windows,
                self.num_bins,
                num_non_zero_indices_in_grid,
                self.Lambda_k.shape[-1],
            ),
            device=self.device,
        )  # init Omega_k

        if self.method == "SQP":
            res = self.SQP(self.Omega_k0)
            Omega_k_opt = res.x.reshape(self.Omega_k0.shape)

        if self.method == "GD_lagrange_multi":
            Omega_k_opt = self.GD_lagrange_multi(iter=itr)

        if self.method == "GD_Deep":
            Omega_k_opt = self.GD_Deep(int(itr))

        if self.method == "SLS":
            Omega_k_opt = self.SLS()

        try:
            utils.plot_loss_tensor(self.reconstruction_loss)
        except:
            print("Failed to plot loss")
        if cheat:
            print("Finished Optimization...Stopping")
            return Bk, Omega_k_opt
        else:
            print("Finished Optimization...Unmixing")
            Sk = torch.zeros(
                (self.num_windows, self.num_bins, self.num_grid_points, Bk.shape[-1]),
                dtype=torch.float32,
            )  # (num windows,num bins,SH Coeff, Window length)
            Sk_temp, Dk = self.unmix_and_smooth(Bk, Omega_k_opt, D_prior)
            print("Finished Unmixing...")
            if mask is not None:
                non_zero_indices_in_grid = torch.nonzero(mask[0, 0, :]).flatten()
                Sk[:, :, non_zero_indices_in_grid, :] = Sk_temp
            else:
                Sk = Sk_temp
            return Sk, Dk
