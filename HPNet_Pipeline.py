import torch
from torch.nn import functional as F
from torch import nn
import math
from sound_field import SoundFieldDataset, create_grid
from optimizer import optimizer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from typing import Union, Optional, List
import utils
import copy


class HPNet_Pipeline:
    def __init__(self,grid_type : str):
        self.grid_type = grid_type
        self.P_th, self.P_ph, self.num_grid_points = create_grid(self.grid_type)

    def set_dataset(self,dataset_path : str):
        self.dataset_path = dataset_path
        self.train_set = SoundFieldDataset(folder_path=dataset_path,dataset_type='train')
        self.val_set = SoundFieldDataset(folder_path=dataset_path,dataset_type='validation')
        self.test_set = SoundFieldDataset(folder_path=dataset_path,dataset_type='test')
        self.input_order = self.train_set.input_order
        self.output_order = self.train_set.output_order

    def init_params(
        self, batch_size: int,
        n_fft: int,
        T: int,
        init_mu: float,
        init_ro: float,
        num_iters : int,
        version : int,
        adam_lr : float = 1e-1,
        net_mode : str = 'DU_simple',
        device : str = 'cpu',
    ):
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.T = T
        self.init_mu = init_mu
        self.init_ro = init_ro
        self.num_iters = num_iters
        self.device = device
        self.adam_lr = adam_lr

        first_order_encoder_mat = utils.create_sh_matrix(
            self.input_order, zen=self.P_th, azi=self.P_ph, type="complex"
        )

        self.du_opt = optimizer(first_order_encoder_mat,
                   alpha=0.1,
                   device=device,
                   save_loss=False,
                   P_th=self.P_th,
                   P_ph=self.P_ph,
                   num_iters=num_iters,
                   num_freq_bins= n_fft//2 + 1,
                   T = T,
                   mu = self.init_mu,
                   ro = self.init_ro,
                   version = version,
                   hyper_parameters= net_mode
                    )

        self.classic_opt = optimizer(first_order_encoder_mat, #version 0
                   alpha=0.1,
                   device=device,
                   save_loss=False,
                   P_th=self.P_th,
                   P_ph=self.P_ph,
                   num_iters=num_iters,
                   num_freq_bins= n_fft//2 + 1,
                   T = T,
                   mu = self.init_mu,
                   ro = self.init_ro,
                   version = 0,
                   hyper_parameters= net_mode
                    )

        self.training_opt = torch.optim.Adam(self.du_opt.parameters(), lr=self.adam_lr)
        # self.du_opt = copy.deepcopy(self.base_optimizer)

    def run_optimizer(
        self,
        from_iter: int,
        to_iter: int,
        low_order_stft: torch.tensor,
        high_order_stft: torch.tensor,
        init_st: Optional[torch.tensor] = None,
        init_lagrange: Optional[torch.tensor] = None,
        progress_per_iter: bool = False,
        disable_tqdm : bool = False,
        classic_model : bool = False, #default is to run on DU model
    ):

        if classic_model:
            optimizer_model = self.classic_opt
        else:
            optimizer_model = self.du_opt

        optimizer_model.batch_preprocess(
            stft_anmt=low_order_stft,
            gt_stft_anmt=high_order_stft,
            init_s_t=init_st,
            init_lagrange_multi=init_lagrange,
        )
        print(f"Init Upscaled Loss : {optimizer_model.upscaled_loss(loss_in_dB=True,s_t=init_st)}")
        with tqdm(total=(to_iter - from_iter), desc="DU_Optimizer", position=1, leave=False,dynamic_ncols=True,disable=disable_tqdm) as pbar:
            for iter in torch.arange(from_iter,to_iter):
                optimizer_model(iter_num = iter, log_losses_per_iter = progress_per_iter)
                pbar.update(1)

        if progress_per_iter:
            optimizer_model.plot_losses_per_iter()
            plt.show()

    def train_iter_group(self,star_iter : int,num_iters_in_group : int, num_epochs : int , train_loader : DataLoader,val_loader : DataLoader):

        for epoch in range(num_epochs):
            epoch_train_loss = 0
            self.du_opt.eval()
            for batch in tqdm(
                train_loader,
                position=0,
                desc=f"Epoch {epoch} Iters {star_iter} - {star_iter + num_iters_in_group}",
            ):
                audio_ids , low_order_stft, high_order_stft = batch
                low_order_stft = low_order_stft.to(self.device)
                high_order_stft = high_order_stft.to(self.device)
                # get init s_t and lagrange multipliers
                self.run_optimizer(
                    star_iter=star_iter,
                    to_iter=star_iter + num_iters_in_group,
                    low_order_stft=low_order_stft,
                    high_order_stft=high_order_stft,
                )

                batch_loss = self.du_opt.upscaled_loss()
                self.training_opt.zero_grad()
                batch_loss.backward()
                self.training_opt.step()

                self.training_opt.step()
                epoch_train_loss += batch_loss.item()
            epoch_train_loss /= len(train_loader)
            train_print = f"Epoch {epoch} loss: {epoch_train_loss}"

            val_print = ""
            if epoch % 50 == 0:
                val_loss = 0
                self.du_opt.eval()
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        audio_ids , low_order_stft, high_order_stft = batch
                        low_order_stft = low_order_stft.to(self.device)
                        high_order_stft = high_order_stft.to(self.device)
                        self.run_optimizer(
                            star_iter=star_iter,
                            to_iter=star_iter + num_iters_in_group,
                            low_order_stft=low_order_stft,
                            high_order_stft=high_order_stft,
                        )
                        val_loss += self.du_opt.upscaled_loss().item()
                val_loss /= len(val_loader)
                val_print = f"| Validation loss: {val_loss}"
            print(train_print + val_print)

    def test_iter_group(self,star_iter : int,num_iters_in_group : int, test_loader : DataLoader):
        test_loss = 0
        self.du_opt.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio_ids , low_order_stft, high_order_stft = batch
                low_order_stft = low_order_stft.to(self.device)
                high_order_stft = high_order_stft.to(self.device)
                self.run_optimizer(
                    star_iter=star_iter,
                    to_iter=star_iter + num_iters_in_group,
                    low_order_stft=low_order_stft,
                    high_order_stft=high_order_stft,
                )
                test_loss += self.du_opt.upscaled_loss().item()
        test_loss /= len(test_loader)
        print(f"Test loss: {test_loss}")

    def train_by_groups(self,iters_per_train : int,num_epochs : int):
        train_loader = DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)
        val_loader = DataLoader(self.val_set,batch_size=self.batch_size,shuffle=True)
        test_loader = DataLoader(self.test_set,batch_size=self.batch_size,shuffle=True)

        first_iter_num = torch.arange(0,self.num_iters,iters_per_train)
        # init lagrange/s_t dict for each iter group
        for start_iter in first_iter_num:
            self.train_iter_group(start_iter,iters_per_train,num_epochs,train_loader,val_loader)
            self.test_iter_group(start_iter,iters_per_train,test_loader)
            # TODO Save s_t and Lagrange multipliers into cpu dict
            # TODO Save model

    def plot_dict(self,source_dict : torch.tensor,index = 50):
        if source_dict.dtype == torch.complex64:
            complex_sparse_stft_dict = source_dict
        else:
            complex_sparse_stft_dict = torch.complex(source_dict[...,:source_dict.shape[-1]//2],source_dict[...,source_dict.shape[-1]//2:])
    
        complex_sparse_stft_dict = (complex_sparse_stft_dict.permute(2,1,0,3).reshape(self.num_grid_points,self.n_fft//2 +1,-1).abs()**2).sum(dim=1).sqrt()
        utils.plot_on_2D(azi=self.P_ph,
                        zen=self.P_th,
                        values=complex_sparse_stft_dict[:,index],
                        title="")
