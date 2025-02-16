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
        P_th,
        P_ph,
        dim_reduction=True,
        constraint_tol=0,
        save_loss=False,
        mag_constraint = False,
        method="GD_lagrange_multi",
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.save_loss = save_loss
        self.Y_p = Y_p
        self.num_SH_coeff = Y_p.shape[1]
        self.num_grid_points = Y_p.shape[0]
        self.alpha = alpha
        self.dim_reduction = dim_reduction
        self.method = method
        self.constraint_tol = constraint_tol
        self.reconstruction_loss = torch.tensor([]).to(self.device)
        self.P_th = P_th
        self.P_ph = P_ph
        self.mag_constraint = mag_constraint

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

    def optimize(self, stft_anmt, iter=1e5,mask=None, mu=1e-1, ro=1e-2, gt_stft_anmt = None,gt_sparse= None): #For paper version
        def grad_dict(s_t, lagrange_multi_k):
            grad = l12_grad(s_t)
            if self.mag_constraint:
                tmp_grad = torch.matmul(reduced_Yp, lagrange_multi_k.to(torch.complex64) )  # No complex conversion
                grad += tmp_grad * (s_t / (torch.abs(s_t) + 1e-10))  # Chain rule for magnitude
            else:
                tmp_grad = torch.matmul(reduced_Yp, torch.complex(lagrange_multi_k[...,:self.T],lagrange_multi_k[...,self.T:]))
                grad += torch.cat((tmp_grad.real,tmp_grad.imag),dim=-1)
            return grad
        
        def l12_grad(s_t):
            grad = s_t / torch.sqrt(
                1e-10 + torch.sum(s_t * torch.conj(s_t), dim=-1, keepdim=True)
            )
            return grad

        def reconstruction_residue(s_t):
            if self.mag_constraint:
                constraint_res = (
                torch.abs(torch.matmul(reduced_Yp.t().conj(), torch.complex(s_t[..., :self.T], s_t[..., self.T:]))) -
                torch.abs(torch.complex(stft_anmt[..., :self.T], stft_anmt[..., self.T:]))
                )
            else:
                constraint_res = torch.matmul(reduced_Yp.t().conj(),torch.complex(s_t[...,:self.T],s_t[...,self.T:])) - torch.complex(stft_anmt[...,:self.T],stft_anmt[...,self.T:])
                constraint_res = torch.cat((constraint_res.real,constraint_res.imag),dim=-1)
            return constraint_res
        
        def up_scaled_loss(gt_stft_anmt):
            constraint_res = torch.matmul(reduced_upscaled_Yp.t().conj(),torch.complex(s_t[...,:self.T],s_t[...,self.T:])) - torch.complex(gt_stft_anmt[...,:self.T],gt_stft_anmt[...,self.T:])
            return torch.cat((constraint_res.real,constraint_res.imag),dim=-1)

        def loss_func(s_t):
            return 10 * torch.log10(
                torch.mean(torch.sum(torch.sqrt(torch.sum(s_t * torch.conj(s_t), dim=-1)),dim=-1)) + 1e-10
            )
        
        def reshape_stft(stft_anmt):
            num_channels = stft_anmt.shape[1]
            num_windows = stft_anmt.shape[0]
            if (num_windows%self.T) > 0:
                stft_anmt = stft_anmt[:-(num_windows%self.T)]
            self.num_windows = stft_anmt.shape[0]
            stft_anmt_1 = stft_anmt.permute(2,1,0)
            stft_anmt_2 = stft_anmt_1.reshape(self.num_bins,num_channels,-1,self.T).permute(2,0,1,3)
            stft_anmt_3 = torch.cat((stft_anmt_2.real, stft_anmt_2.imag), dim=-1)
            
            return stft_anmt_3

        if mask is None:
            self.mask = torch.arange(self.num_grid_points).to(self.device)
        else:
            self.mask = mask

        self.num_windows, self.num_channels, self.num_bins = stft_anmt.shape
        stft_anmt = reshape_stft(stft_anmt)
        if gt_stft_anmt is not None:
            reduced_upscaled_Yp = utils.create_sh_matrix(int(torch.sqrt(torch.tensor(gt_stft_anmt.shape[1])) - 1) , zen=self.P_th, azi=self.P_ph,type='complex')[self.mask,:].to(self.device)
            gt_stft_anmt = reshape_stft(gt_stft_anmt).to(self.device)
        self.N = stft_anmt.shape[0]
        lagrange_multi_t = torch.zeros(
            (*stft_anmt.shape[:-1],self.T * (1 if self.mag_constraint else 2)),
            dtype=stft_anmt.dtype,
        ).to(self.device)
        s_t = 0 *torch.randn(
            self.N,
            self.num_bins,
            len(self.mask),
            self.T * 2,
            dtype=stft_anmt.dtype,
        ).to(self.device)
        self.L12_loss = torch.tensor([]).to(self.device)
        self.reconstruction_loss = torch.tensor([]).to(self.device)
        self.gt_upscale_loss = torch.tensor([]).to(self.device)
        self.sparse_dict_loss = torch.tensor([]).to(self.device)
        reduced_Yp = self.Y_p[self.mask,:].to(self.device)
        for iii in tqdm(range(int(iter))):

            if self.method == "GD_lagrange_multi":
                grad_s = grad_dict(s_t, lagrange_multi_t)
                recon_residue = reconstruction_residue(s_t) #this is the grad for the lagrange multipliers relative to lambda
                s_t -= mu * grad_s
                lagrange_multi_t += ro * recon_residue

            if self.method == "GD_regularization":
                recon_residue = reconstruction_residue(s_t) #this is real/imag stacked in last dimension
                tmp_grad = torch.matmul(reduced_Yp,torch.complex(recon_residue[...,:self.T],recon_residue[...,self.T:]))
                grad_s = 2*torch.cat((tmp_grad.real,tmp_grad.imag),dim=-1) + ro*l12_grad(s_t)
                s_t -= mu * grad_s

            self.L12_loss = torch.cat(
                (self.L12_loss, loss_func(s_t).unsqueeze(0)),
                dim=0,
            )
            self.reconstruction_loss = torch.cat(
                (self.reconstruction_loss,10*torch.log10((recon_residue**2).sum().unsqueeze(0)))
            )
            if gt_stft_anmt is not None:
                self.gt_upscale_loss = torch.cat(
                (self.gt_upscale_loss,10*torch.log10((up_scaled_loss(gt_stft_anmt)**2).sum().unsqueeze(0)))
            )
            if gt_sparse is not None:
                pass

        plt.figure()
        plt.plot(self.L12_loss.cpu().detach())
        plt.xlabel("Iterations")
        plt.ylabel("Loss [dB]")
        plt.title(f"L12 Loss mu = {mu:.1e} ro = {ro:.1e} T = {self.T}")

        plt.figure()
        plt.plot(self.reconstruction_loss.cpu().detach())
        plt.xlabel("Iterations")
        plt.ylabel("Loss [dB]")
        plt.title(f"Input Reconstruction Loss mu = {mu:.1e} ro = {ro:.1e} T = {self.T}")


        if gt_stft_anmt is not None:
            plt.figure()
            plt.plot(self.gt_upscale_loss.cpu().detach())
            plt.xlabel("Iterations")
            plt.ylabel("Loss [dB]")
            plt.title(f"Upscale L2 Loss mu = {mu:.1e} ro = {ro:.1e} T = {self.T}")

        if gt_sparse is not None:
            plt.figure()
            plt.plot(self.sparse_dict_loss.cpu().detach())
            plt.xlabel("Iterations")
            plt.ylabel("Loss [dB]")
            plt.title("Sparse Dict L2 Loss")

        expanded_st = torch.zeros(self.num_windows,
            self.num_grid_points,
            self.num_bins * 2,
            dtype=stft_anmt.dtype,)
        #go for certainty:
        tmp = s_t.cpu()
        tmp = torch.complex(tmp[...,:self.T],tmp[...,self.T:]).permute(2,1,0,3).reshape(len(self.mask),self.num_bins,-1).permute(2,0,1)
        tmp = torch.cat((tmp.real,tmp.imag),dim=-1)
        expanded_st[:,self.mask,:] = tmp

        return expanded_st

    # def optimize(self, Bk, itr=1e5, mask=None, D_prior=None, cheat=False):
    #     if isinstance(Bk, tuple):
    #         # cheap hack for MP
    #         tmp = Bk
    #         Bk = tmp[0]
    #         mask = tmp[1]
    #         D_prior = tmp[2]
    #     self.num_windows, self.num_bins = Bk.shape[:2]

    #     # Dim Reduction
    #     if mask is None:
    #         mask = torch.ones(
    #             (self.num_windows, self.num_bins, self.Y_p.shape[1]),
    #             dtype=bool,
    #             device=self.device,
    #         )
    #     else:
    #         if mask.shape[0] == 1 and mask.shape[1] == 1:
    #             mask = torch.broadcast_to(
    #                 mask, (self.num_windows, self.num_bins, mask.shape[-1])
    #             ).to(self.device)

    #     num_non_zero_indices_in_grid = torch.count_nonzero(
    #         mask[0, 0, :]
    #     )  # ASSUMES MASK DOESNT CHANGE OVER TIME
    #     self.mask = mask
    #     self.Uk, Lk, Vkt = torch.linalg.svd(Bk, full_matrices=False)
    #     self.Uk = self.Uk.to(self.device)
    #     self.Lambda_k = torch.diag_embed(Lk).to(self.device)
    #     self.Omega_k0 = torch.zeros(
    #         (
    #             self.num_windows,
    #             self.num_bins,
    #             num_non_zero_indices_in_grid,
    #             self.Lambda_k.shape[-1],
    #         ),
    #         device=self.device,
    #     )  # init Omega_k

    #     if self.method == "SQP":
    #         res = self.SQP(self.Omega_k0)
    #         Omega_k_opt = res.x.reshape(self.Omega_k0.shape)

    #     if self.method == "GD_lagrange_multi":
    #         Omega_k_opt = self.GD_lagrange_multi(iter=itr)

    #     if self.method == "GD_Deep":
    #         Omega_k_opt = self.GD_Deep(int(itr))

    #     if self.method == "SLS":
    #         Omega_k_opt = self.SLS()

    #     try:
    #         utils.plot_loss_tensor(self.reconstruction_loss)
    #     except:
    #         print("Failed to plot loss")
    #     if cheat:
    #         print("Finished Optimization...Stopping")
    #         return Bk, Omega_k_opt
    #     else:
    #         print("Finished Optimization...Unmixing")
    #         Sk = torch.zeros(
    #             (self.num_windows, self.num_bins, self.num_grid_points, Bk.shape[-1]),
    #             dtype=torch.float32,
    #         )  # (num windows,num bins,SH Coeff, Window length)
    #         Sk_temp, Dk = self.unmix_and_smooth(Bk, Omega_k_opt, D_prior)
    #         print("Finished Unmixing...")
    #         if mask is not None:
    #             non_zero_indices_in_grid = torch.nonzero(mask[0, 0, :]).flatten()
    #             Sk[:, :, non_zero_indices_in_grid, :] = Sk_temp
    #         else:
    #             Sk = Sk_temp
    #         return Sk, Dk
