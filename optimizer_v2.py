from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
from sound_field import SoundField
import utils


class optimizer_v2(nn.Module):
    def __init__(
        self,
        sound_field: SoundField,
        Y_p: torch.tensor,
        alpha: int,
        num_iters: int,
        device: str = "cpu",
        deep_unfolded: bool = False,
    ):
        super().__init__()
        self.device = device
        self.sound_field = sound_field
        self.Y_p = Y_p.to(self.device)
        self.num_SH_coeff = Y_p.shape[0]
        self.num_grid_points = Y_p.shape[1]
        self.num_iters = int(num_iters)
        self.alpha = alpha
        self.deep_unfolded = deep_unfolded

    def low_dim_regularization(self, Omega_k):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        res = torch.sum(torch.sqrt(torch.sum(Omega_k**2, dim=1)))
        return res

    def low_dim_reconstruction_loss(self, Omega_k, Y_p=None):
        Omega_k = Omega_k.reshape(self.Omega_k0.shape)
        if Y_p is None:
            Y_p = self.Y_p
        low_dim_reconstruction_loss = (
            torch.norm(
                (
                    torch.matmul(Y_p, Omega_k) - torch.matmul(self.Uk, self.Lambda_k)
                ).reshape(self.num_windows, self.num_bins, -1),
                p=2,
            )
            ** 2
        )
        return low_dim_reconstruction_loss

    def full_low_dim_loss_grad(self, Omega_k: torch.tensor, iter: int):

        # Regularization Grad
        low_dim_regularization_grad = (
            self.learned_lambda[iter]
            * Omega_k
            / (torch.sqrt(torch.sum(Omega_k**2, dim=-1))[..., None])
        )

        # Reconstruction Grad
        transposed_Yp = self.reduced_Yp.permute(0, 1, 3, 2)
        broadcasted_transposed_Yp = torch.broadcast_to(
            transposed_Yp,
            (
                self.num_windows,
                self.num_bins,
                transposed_Yp.shape[-2],
                transposed_Yp.shape[-1],
            ),
        )
        reconstruction_grad = 2 * torch.matmul(
            broadcasted_transposed_Yp,
            (
                torch.matmul(self.reduced_Yp, Omega_k)
                - torch.matmul(self.Uk, self.Lambda_k)
            ),
        )
        return reconstruction_grad + low_dim_regularization_grad

    def pre_processing(self, Bk, mask):
        self.num_windows, self.num_bins = Bk.shape[:2]
        self.init_weights()
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
        self.Y_p = torch.broadcast_to(
            self.Y_p, (self.num_windows, self.num_bins, *self.Y_p.shape)
        )

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

    def perform_optimization(self):
        non_zero_indices_in_grid = torch.nonzero(self.mask[0, 0, :]).to(self.device)
        self.reduced_Yp = self.Y_p[:, :, :, non_zero_indices_in_grid].reshape(
            self.num_windows, self.num_bins, self.num_SH_coeff, -1
        )  # TODO Currently assumes mask doesnt change over time

        Omega_k = torch.randn(*self.Omega_k0.shape).to(self.device)
        self.low_dim_reconstruction_loss_per_iter = list()
        for iter in tqdm(range(self.num_iters)):
            Omega_k -= self.learned_mu[iter] * self.full_low_dim_loss_grad(
                Omega_k, iter
            )

            self.low_dim_reconstruction_loss_per_iter.append(
                10
                * torch.log10(torch.sum(self.low_dim_reconstruction_loss(Omega_k)))
                .cpu()
                .detach()
            )
        return Omega_k

    def post_processing(self, Bk, Omega_k_opt, D_prior):
        #Placeholder for the sparse dictionary
        Sk = torch.zeros(
            (self.num_windows, self.num_bins, self.num_grid_points, Bk.shape[-1]),
            dtype=torch.float32,
        )  # (num windows,num bins,SH Coeff, Window length)

        #Unmix & Smooth
        Dk = torch.matmul(
            Omega_k_opt,
            torch.linalg.pinv(torch.matmul(self.Uk, self.Lambda_k)),
        ).cpu()
        if D_prior is not None:
            Dk = self.alpha * Dk + (1 - self.alpha) * D_prior
        Sk_gpu = torch.matmul(Dk, Bk)
        Sk_cpu = Sk_gpu.cpu()
        Dk_cpu = Dk.cpu()
        non_zero_indices_in_grid = torch.nonzero(self.mask[0, 0, :]).flatten()
        Sk[:, :, non_zero_indices_in_grid, :] = Sk_cpu

        return Sk, Dk_cpu

    def init_weights(self):
        self.learned_mu = nn.Parameter(
            torch.full(
                (self.num_iters, 1, self.num_bins, 1, 1), 1e-3, device=self.device
            ),
            requires_grad=self.deep_unfolded,
        )
        self.learned_lambda = nn.Parameter(
            torch.full(
                (self.num_iters, 1, self.num_bins, 1, 1),
                self.alpha,
                device=self.device,
            ),
            requires_grad=self.deep_unfolded,
        )

    def forward(self, mask, upscale_order=3, preprocessing=True):

        if preprocessing:
            Bk = self.sound_field.windowed_anm_t.permute(
                0, 1, 3, 2
            )  # turn to (window,band,SH_coeff,time)
            if mask is not None:
                mask_matrix = mask[None, None, ...]
            else:
                mask_matrix = None
            self.pre_processing(Bk, mask_matrix)

        #Note2Self : If we do optimization per band/window the post processing and opt need to come together
        #the rest is only when we are all down with the optimization
        opt_Omega_K = self.perform_optimization()
        self.sound_field.sparse_dict_subbands, Dk_cpu = self.post_processing(
            Bk, opt_Omega_K, D_prior=None
        )


        # sum of all subbands
        self.sound_field.s_windowed = torch.sum(
            self.sound_field.sparse_dict_subbands, axis=1
        )

        self.sound_field.s_dict = self.sound_field.s_windowed.permute(1, 0, 2).reshape(
            self.sound_field.num_grid_points,
            self.sound_field.window_length * self.sound_field.num_windows,
        )
        Y_p_tag = utils.create_sh_matrix(
            upscale_order, zen=self.sound_field.P_th, azi=self.sound_field.P_ph
        )  # upscaling matrix
        self.sound_field.anm_upscaled = torch.matmul(Y_p_tag, self.sound_field.s_dict)
        return self.sound_field.anm_upscaled
