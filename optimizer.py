from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
import utils
from HyperParaNet import HPNet
from typing import Optional,Union,List


class optimizer(nn.Module):
    def __init__(
        self,
        Y_p,
        alpha,
        P_th,
        P_ph,
        dim_reduction=True,
        constraint_tol=0,
        num_freq_bins : int = 129,
        T : int = 10,
        mu : float = 1e-2,
        ro : float = 1e-3,
        version : int = 0,
        save_loss=False,
        opt_method="GD_lagrange_multi",
        device="cpu",
        hyper_parameters = 'classic'
    ):
        super().__init__()
        self.device = device
        self.save_loss = save_loss
        self.Y_p = Y_p
        self.ambi_channels = Y_p.shape[1]
        self.num_grid_points = Y_p.shape[0]
        self.alpha = alpha
        self.dim_reduction = dim_reduction
        self.opt_method = opt_method
        self.constraint_tol = constraint_tol
        self.reconstruction_loss = torch.tensor([]).to(self.device)
        self.P_th = P_th
        self.P_ph = P_ph
        self.T = T
        self.num_freq_bins = num_freq_bins
        self.hyper_parameters = hyper_parameters
        self.version = version
        self.init_mu = mu
        self.init_ro = ro
        print(f"Optimization Method : {self.opt_method} ")



    def init(self, stft_anmt,gt_stft_anmt = None, mask = None):

        if mask is None:
            self.mask = torch.arange(self.num_grid_points).to(self.device)
        else:
            self.mask = mask

        self.L12_loss = torch.tensor([]).to(self.device)
        self.reconstruction_loss = torch.tensor([]).to(self.device)
        self.gt_upscale_loss = torch.tensor([]).to(self.device)
        self.sparse_dict_loss = torch.tensor([]).to(self.device)
        self.reduced_Yp = self.Y_p.to(self.device)[self.mask,:]

        _, self.num_channels, self.num_bins = stft_anmt.shape[-3:] #update num windows after mod(self.T)
        self.stft_anmt = self.reshape_stft(stft_anmt).to(self.device)
        self.complex_input_order =  torch.complex(self.stft_anmt[...,:self.T],self.stft_anmt[...,self.T:])
        # energy= (torch.abs(complex_input_order)**2).sum().sqrt() + 1e-8# torch.norm(complex_gt_stft_anmt,p=2,dim=-1,keepdim=True).sum(1,keepdim=True) + 1e-8
        # complex_input_order /= energy

    
        self.N = self.stft_anmt.shape[-4]
        if gt_stft_anmt is not None:
            self.gt_stft_anmt = self.reshape_stft(gt_stft_anmt).to(self.device)
            self.reduced_upscaled_Yp = utils.create_sh_matrix(int(torch.sqrt(torch.tensor(self.gt_stft_anmt.shape[3])) - 1) , zen=self.P_th, azi=self.P_ph,type='complex').to(self.device)[self.mask,:]
            self.complex_gt_stft_anmt = torch.complex(self.gt_stft_anmt[...,:self.T],self.gt_stft_anmt[...,self.T:])
            # energy= (torch.abs(complex_gt_stft_anmt)**2).sum().sqrt() + 1e-8# torch.norm(complex_gt_stft_anmt,p=2,dim=-1,keepdim=True).sum(1,keepdim=True) + 1e-8
            # complex_gt_stft_anmt /= energy
        else:
            self.gt_stft_anmt = None
            self.complex_gt_stft_anmt = None


        self.lagrange_multi_t = torch.zeros(
            (self.stft_anmt.shape),
            dtype=self.stft_anmt.dtype,
        ).to(self.device)

        if self.stft_anmt.dim() == 5:
            self.s_t = torch.zeros(
                self.stft_anmt.shape[0], # num batches
                self.N,
                self.num_bins,
                len(self.mask),
                self.T * 2,
                dtype=self.stft_anmt.dtype,
            ).to(self.device)
        else:
             self.s_t = torch.zeros(
                self.N,
                self.num_bins,
                len(self.mask),
                self.T * 2,
                dtype=self.stft_anmt.dtype,
            ).to(self.device)

        self.HP_model = HPNet(
            self.hyper_parameters,
            self.num_grid_points,
            self.num_iters,
            self.num_freq_bins,
            self.T,
            self.ambi_channels,
            self.init_mu,
            self.init_ro,
            self.version,
        ).to(self.device)



    def reshape_stft(self,stft_anmt):
        if stft_anmt.dim() == 4:
            batches = stft_anmt.shape[0]
        else:
            batches = None
        curr_num_windows = stft_anmt.shape[-3] # 
        if (curr_num_windows % self.T) > 0:
            stft_anmt = stft_anmt[:,:-(curr_num_windows%self.T)]
        self.num_windows = stft_anmt.shape[-3]
        if batches is not None:
            stft_anmt_2 = stft_anmt.permute(-4,-1,-2,-3).reshape(batches,self.num_bins,stft_anmt.shape[-2],-1,self.T).permute(-5,-2,-4,-3,-1)
        else:
            stft_anmt_2 = stft_anmt.permute(-1,-2,-3).reshape(self.num_bins,stft_anmt.shape[-2],-1,self.T).permute(-2,-4,-3,-1)

        stft_anmt_3 = torch.cat((stft_anmt_2.real, stft_anmt_2.imag), dim=-1)

        return stft_anmt_3


    def grad_dict(self,recon_residue=None):
        grad = self.l12_grad()

        tmp_grad = torch.matmul(self.reduced_Yp, torch.complex(self.lagrange_multi_t[...,:self.T],self.lagrange_multi_t[...,self.T:]))
        grad += torch.cat((tmp_grad.real,tmp_grad.imag),dim=-1)

        if recon_residue is not None:
            assert False, 'ALM not in use'
            penalty_term = torch.matmul(self.reduced_Yp, torch.complex(recon_residue[...,:self.T], recon_residue[...,self.T:]))
            grad += ro * 2 * torch.cat((penalty_term.real,penalty_term.imag),dim=-1) 

        return grad
    
    def l12_grad(self):
        grad = self.s_t / torch.sqrt(
            1e-10 + torch.sum(self.s_t * torch.conj(self.s_t), dim=-1, keepdim=True)
        )
        return grad
    
    def reconstruction_residue(self):
        complex_input_order_est = torch.matmul(self.reduced_Yp.t().conj(),torch.complex(self.s_t[...,:self.T],self.s_t[...,self.T:]))
        constraint_res = complex_input_order_est - self.complex_input_order
        constraint_res = torch.cat((constraint_res.real,constraint_res.imag),dim=-1)
        return constraint_res,complex_input_order_est


    def input_order_loss(self,complex_input_order_est):
        denom = torch.norm(self.complex_input_order, p=2, dim = (-2,-1))**2
        nom = torch.norm(complex_input_order_est - self.complex_input_order, p=2, dim = (-2,-1))**2
        loss = (nom).mean()
        log_loss = 10*torch.log10(loss).unsqueeze(0)
        return log_loss

    def upscaled_loss(self,loss_in_dB : bool = False):
        if  self.complex_gt_stft_anmt is None:
            return None
        upscaled_est = torch.matmul(self.reduced_upscaled_Yp.t().conj(),torch.complex(self.s_t[...,:self.T],self.s_t[...,self.T:]))
        constraint_res = upscaled_est - self.complex_gt_stft_anmt
        denom = torch.norm(self.complex_gt_stft_anmt, p=2, dim = (-2,-1))**2
        nom = torch.norm(constraint_res, p=2, dim = (-2,-1))**2
        loss = (nom).mean()
        if loss_in_dB:
            loss_dB = 10*torch.log10(loss).unsqueeze(0)
            return loss_dB
        else:
            return loss
    
    def l12_loss(self):
        return 10 * torch.log10(
            torch.mean(torch.sum(torch.sqrt(torch.sum(self.s_t * torch.conj(self.s_t), dim=-1)),dim=-1)) + 1e-10
        )

    
    def forward(self,iter_num : int, log_losses_per_iter : bool = True):

        if self.opt_method == "GD_lagrange_multi":
            recon_residue,complex_input_order_est = self.reconstruction_residue() #this is the grad for the lagrange multipliers relative to lambda
            grad_s = self.grad_dict()
            mu,ro = self.HP_model(iter_num) 
            # v = 0.9 * v + grad_s
            self.s_t = self.s_t -  mu.exp() * grad_s
            self.lagrange_multi_t = self.lagrange_multi_t + ro.exp() * recon_residue

        if self.opt_method == "GD_AUG_lagrange_multi":
            recon_residue,complex_input_order_est = self.reconstruction_residue(s_t) #this is the grad for the lagrange multipliers relative to lambda
            grad_s = self.grad_dict(recon_residue)
            self.s_t -= mu * grad_s
            self.lagrange_multi_t += ro * recon_residue

        if self.opt_method == "GD_regularization":
            recon_residue,complex_input_order_est = self.reconstruction_residue() #this is real/imag stacked in last dimension
            tmp_grad = torch.matmul(self.reduced_Yp,torch.complex(recon_residue[...,:self.T],recon_residue[...,self.T:]))
            grad_s = 2*torch.cat((tmp_grad.real,tmp_grad.imag),dim=-1) + ro*self.l12_grad()
            self.s_t -= mu * grad_s


        if log_losses_per_iter: #currently, dont keep track if we are not about to plot - and we only plot when we have batch = 1 (code doesnt handle multiple batches)
            self.L12_loss = torch.cat(
                (self.L12_loss, self.l12_loss().unsqueeze(0)/self.N),
                dim=0,
            )
            self.reconstruction_loss = torch.cat(
                (self.reconstruction_loss,self.input_order_loss(complex_input_order_est)),
            )

            if self.gt_stft_anmt is not None:
                self.gt_upscale_loss = torch.cat(
                (self.gt_upscale_loss,self.upscaled_loss(loss_in_dB=True))
            )

    def get_st(self):
        if self.num_grid_points == len(self.mask):
            return self.s_t
        else:
            #We were using a mask, so expand the dictionary back
            expanded_st = torch.zeros(self.num_windows,
                self.num_grid_points,
                self.num_bins * 2,
                dtype=self.stft_anmt.dtype,).to(self.device)
            tmp = torch.complex(self.s_t[...,:self.T],self.s_t[...,self.T:]).permute(2,1,0,3).reshape(len(self.mask),self.num_bins,-1).permute(2,0,1)
            tmp = torch.cat((tmp.real,tmp.imag),dim=-1)
            expanded_st[:,self.mask,:] = tmp
            return expanded_st
            


    def plot_losses_per_iter(self,mu : Optional[float] = None, ro : Optional[float] = None ,from_iter : int = 0 ):
        tit = ""
        if mu is not None and ro is not None and self.hyper_parameters == 'classic':
            tit = f'mu = {mu:.1e} ro = {ro:.1e}'
        parameters_title = f"{self.hyper_parameters} {tit} T = {self.T}"
        plt.figure()
        plt.plot(self.L12_loss[from_iter:].cpu().detach())
        plt.xlabel("Iterations")
        plt.ylabel("Loss [dB]")
        plt.title(f"L12 Loss {parameters_title}")

        plt.figure()
        plt.plot(self.reconstruction_loss[from_iter:].cpu().detach())
        plt.xlabel("Iterations")
        plt.ylabel("Loss [dB]")
        plt.title(f"Input Reconstruction Loss {parameters_title}")

        if self.gt_stft_anmt is not None:
            plt.figure()
            plt.plot(self.gt_upscale_loss[from_iter:].cpu().detach())
            plt.xlabel("Iterations")
            plt.ylabel("Loss [dB]")
            plt.title(f"Upscale L2 Loss {parameters_title}")

class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class HyperNet(nn.Module):
    def __init__(self,mode : str = None):
        super().__init__()
