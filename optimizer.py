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

    def optimize(self, stft_anmt, iter=1e5,mask=None, mu=1e-1, ro=1e-2, gt_stft_anmt = None,gt_sparse= None):
        def grad_dict(s_t, lagrange_multi_k,recon_residue=None):
            grad = l12_grad(s_t)

            tmp_grad = torch.matmul(reduced_Yp, torch.complex(lagrange_multi_k[...,:self.T],lagrange_multi_k[...,self.T:]))
            grad += torch.cat((tmp_grad.real,tmp_grad.imag),dim=-1)

            if recon_residue is not None:
                penalty_term = torch.matmul(reduced_Yp, torch.complex(recon_residue[...,:self.T], recon_residue[...,self.T:]))
                grad += ro * 2 * torch.cat((penalty_term.real,penalty_term.imag),dim=-1) 

            return grad
        
        def l12_grad(s_t):
            grad = s_t / torch.sqrt(
                1e-10 + torch.sum(s_t * torch.conj(s_t), dim=-1, keepdim=True)
            )
            return grad

        def reconstruction_residue(s_t):
            complex_input_order_est = torch.matmul(reduced_Yp.t().conj(),torch.complex(s_t[...,:self.T],s_t[...,self.T:]))
            constraint_res = complex_input_order_est - complex_input_order
            constraint_res = torch.cat((constraint_res.real,constraint_res.imag),dim=-1)
            return constraint_res,complex_input_order_est
        
        def input_order_loss(complex_input_order_est,complex_input_order):
            denom = torch.norm(complex_input_order, p=2, dim = (2,3))**2
            nom = torch.norm(complex_input_order_est - complex_input_order, p=2, dim = (2,3))**2
            loss = (nom).mean()
            log_loss = 10*torch.log10(loss).unsqueeze(0)
            return log_loss
            
        def up_scaled_loss(s_t):
            upscaled_est = torch.matmul(reduced_upscaled_Yp.t().conj(),torch.complex(s_t[...,:self.T],s_t[...,self.T:]))
            constraint_res = upscaled_est - complex_gt_stft_anmt
            denom = torch.norm(complex_gt_stft_anmt, p=2, dim = (2,3))**2
            nom = torch.norm(constraint_res, p=2, dim = (2,3))**2
            loss = (nom).mean()
            log_loss = 10*torch.log10(loss).unsqueeze(0)
            return log_loss

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
            (stft_anmt.shape),
            dtype=stft_anmt.dtype,
        ).to(self.device)
        s_t = 0 *torch.randn(
            self.N,
            self.num_bins,
            len(self.mask),
            self.T * 2,
            dtype=stft_anmt.dtype,
        ).to(self.device)

        complex_gt_stft_anmt = torch.complex(gt_stft_anmt[...,:self.T],gt_stft_anmt[...,self.T:])
        # energy_per_dir= torch.norm(complex_gt_stft_anmt,p=2,dim=-1,keepdim=True).sum(1,keepdim=True) + 1e-8
        # complex_gt_stft_anmt /= energy_per_dir

        complex_input_order =  torch.complex(stft_anmt[...,:self.T],stft_anmt[...,self.T:])
        # energy_per_dir= torch.norm(complex_input_order,p=2,dim=-1,keepdim=True).sum(1,keepdim=True) + 1e-8
        # complex_input_order /= energy_per_dir
        


        self.L12_loss = torch.tensor([]).to(self.device)
        self.reconstruction_loss = torch.tensor([]).to(self.device)
        self.gt_upscale_loss = torch.tensor([]).to(self.device)
        self.sparse_dict_loss = torch.tensor([]).to(self.device)
        reduced_Yp = self.Y_p[self.mask,:].to(self.device)
        for iii in tqdm(range(int(iter))):

            if self.method == "GD_lagrange_multi":
                recon_residue,complex_input_order_est = reconstruction_residue(s_t) #this is the grad for the lagrange multipliers relative to lambda
                grad_s = grad_dict(s_t, lagrange_multi_t)
                s_t -= mu * grad_s
                lagrange_multi_t += ro * recon_residue


            if self.method == "GD_AUG_lagrange_multi":
                recon_residue,complex_input_order_est = reconstruction_residue(s_t) #this is the grad for the lagrange multipliers relative to lambda
                grad_s = grad_dict(s_t, lagrange_multi_t,recon_residue)
                s_t -= mu * grad_s
                lagrange_multi_t += ro * recon_residue

            if self.method == "GD_regularization":
                recon_residue,complex_input_order_est = reconstruction_residue(s_t) #this is real/imag stacked in last dimension
                tmp_grad = torch.matmul(reduced_Yp,torch.complex(recon_residue[...,:self.T],recon_residue[...,self.T:]))
                grad_s = 2*torch.cat((tmp_grad.real,tmp_grad.imag),dim=-1) + ro*l12_grad(s_t)
                s_t -= mu * grad_s
            
            self.L12_loss = torch.cat(
                (self.L12_loss, loss_func(s_t).unsqueeze(0)/self.N),
                dim=0,
            )
            self.reconstruction_loss = torch.cat(
                (self.reconstruction_loss,input_order_loss(complex_input_order_est,complex_input_order)),
            )
            
            if gt_stft_anmt is not None:
                self.gt_upscale_loss = torch.cat(
                (self.gt_upscale_loss,up_scaled_loss(s_t))
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
    

class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class HyperNet(nn.Module):
    def __init__(self,mode : str = None):
        super().__init__()
    
