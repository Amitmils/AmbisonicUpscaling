import utils
import math
import multiprocessing as mp
from typing import List, Union, Optional
import scipy.io
from tqdm import tqdm
import os
import torch
from datetime import datetime
from py_bank.filterbanks_pytorch import EqualRectangularBandwidth
from signal_info import signal_info
from optimizer import optimizer
import sounddevice as sd
from torch.utils.data import Dataset
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import ast
import zipfile


LEBEDEV_GRID_PATH = "Lebvedev2702.mat"
LEBEDEV = "lebedev"
POINTS_162 = "162_points"


def create_grid(grid_type):
    if grid_type == LEBEDEV:
        num_grid_points = 2702  # P
        lebedev = scipy.io.loadmat("Lebvedev2702.mat")
        P_th = torch.tensor(lebedev["th"].reshape(-1))  # rad
        P_ph = torch.tensor(lebedev["ph"].reshape(-1))  # rad
        P_ph = (P_ph + torch.pi) % (2 * torch.pi) - torch.pi  # wrap angles to [-pi,pi]
    elif grid_type == POINTS_162:
        num_grid_points = 162  # P
        points = utils.generate_sphere_points(162, plot=False)
        P_th = points[:, 1]
        P_ph = points[:, 2]
    else:
        raise ValueError(f"Unknown grid type {grid_type}")
    return P_th, P_ph, num_grid_points


def divide_to_subbands(
    anm_t: torch.tensor,
    num_bins: int,
    downsample: int = 1,
    low_filter_center_freq: int = 1,
    sr: int = 16000,
) -> torch.tensor:
    # signal is size [num_samples,(ambi Order+1)^2]
    anm_t = anm_t[::downsample].squeeze()
    # assert downsample == 1, "Downsample is not supported yet"
    sr = sr / downsample  # FIX - if downsample > 1 we need a LPF

    if num_bins == 1:
        anm_t_subbands = anm_t[None, ...]
    else:
        num_bins -= 2  # for perfect reconstruction (first bin is low pass and last bin is high pass)
        high_filter_center_freq = sr / 2  # centre freq. of highest filter
        num_samples, num_coeff = anm_t.shape  # filter bank length
        erb_bank = EqualRectangularBandwidth(
            num_samples,
            sr,
            num_bins,
            low_filter_center_freq,
            high_filter_center_freq,
        )
        anm_t_subbands = torch.zeros(
            (num_bins + 2, num_samples, num_coeff), device=anm_t.device
        )  # num_bins + low and high for perfect reconstruction  | filter_length = num of SH coeff | num_samples = t
        for coeff in range(num_coeff):
            erb_bank.generate_subbands(anm_t[:, coeff])
            anm_t_subbands[:, :, coeff] = (
                torch.tensor(erb_bank.subbands).clone().detach()
            )

    # [pass band k,t,SH_coeff]
    return anm_t_subbands


def divide_to_time_windows(
    anm_t_subbands: torch.tensor, window_length: int, max_num_windows: int
) -> torch.tensor:
    # signal is size [band pass k ,time samples,(ambi Order+1)^2]
    num_samples = anm_t_subbands.shape[1]

    # Pad the tensor
    padding = (0, 0, 0, window_length - num_samples % window_length)
    anm_t_padded = torch.nn.functional.pad(
        anm_t_subbands, padding, mode="constant", value=0
    )

    # Split the tensor into windows (Window,Freq Bin,SH Coeff,Time)
    windowed_anm_t = anm_t_padded.unfold(
        dimension=1, step=window_length, size=window_length
    ).permute(1, 0, 3, 2)

    windowed_anm_t = windowed_anm_t[:max_num_windows]
    return windowed_anm_t


class SoundField:
    def __init__(self, device: torch.device = torch.device("cpu"),maximum_seconds = 5) -> None:
        self.device = device
        self.maximum_seconds = maximum_seconds

    def _build_joint_soundfield(self,order,SH_type,n_fft,normalize_signals):
        max_length = 0
        self.anm_f_list = list()
        self.sources_coords = list()
        for curr_sig in self.signals:
            self.sources_coords.append(
                (math.degrees(curr_sig.th), math.degrees(curr_sig.ph))
            )
            anm_f, s_f, _, y = utils.encode_signal(
                curr_sig,
                order,
                plot=False,
                type=SH_type,
                n_fft = n_fft,
            )
            num_frames = anm_f.shape[-1]
            max_length = max(max_length, anm_f.shape[-1])
            self.anm_f_list.append(anm_f)

        # combine all signals
        max_frames = int(2*(self.maximum_seconds * self.sr)/n_fft) # seconds * sr / (nfft/2)  --> Assumees 50% overlap
        max_length = min(max_length,max_frames)
        total_anm_f = torch.zeros(((order+1)**2,n_fft//2 +1,max_length),dtype=self.anm_f_list[0].dtype)
        for i in range(len(self.anm_f_list)):
            num_frames = self.anm_f_list[i].shape[-1]
            if num_frames < max_length:
                total_anm_f += torch.nn.functional.pad(
                    self.anm_f_list[i],
                    (0, max_length - num_frames),
                )
            else:
                total_anm_f += self.anm_f_list[i][...,:max_length]

        if normalize_signals:
            energy= (torch.abs(total_anm_f)**2).sum().sqrt() + 1e-8
            total_anm_f /= energy

        return total_anm_f

    def load(self,
             anm_f_input,
             anm_f_output,
             csv_meta_data_path :str,
             id : int,
             debug : bool = True,
             SH_type: str = "complex",
             grid_type: str = LEBEDEV):
        meta_data = pd.read_csv(csv_meta_data_path)
        audio_meta_data = meta_data.loc[id]
        self.sr = audio_meta_data['sr']
        self.P_th, self.P_ph, self.num_grid_points = create_grid(grid_type)
        self.sources_coords = ast.literal_eval(audio_meta_data['DOAs'])
        self.input_order = int(math.sqrt(anm_f_input.shape[0])-1)
        self.output_order = int(math.sqrt(anm_f_output.shape[0])-1)


        if debug:
            # project first time sample on 162 points
            for order,anm_f in zip([self.input_order,self.output_order],[anm_f_input,anm_f_output]):
                t = 10
                Y_p = utils.create_sh_matrix(
                    order, zen=self.P_th, azi=self.P_ph, type=SH_type
                )
                projected_values = (torch.abs((Y_p @ anm_f[:,:,t] ))**2).sum(dim=1)
                utils.plot_on_2D(
                    azi=self.P_ph,
                    zen=self.P_th,
                    values=projected_values,
                    title=f"Encoded Signal N={order}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
                )

        


    def load_matlab(
        self,
        file_path: str,
        n_fft: int = 1024,
        grid_type: str = LEBEDEV,
        SH_type: str = "complex",
        debug: bool = False,
        t : int = 10,
    ) -> None:
        
        data = loadmat(file_path)
        anm_t_list = data['anm_t_list'].squeeze()
        self.sr = data['fs'].item()
        self.P_th, self.P_ph, self.num_grid_points = create_grid(grid_type)
        self.has_mask = False
        self.sources_coords = [[math.degrees(data['src_theta']),math.degrees(data['src_phi'])]]
        hop_length = n_fft//2   # Hop length (stride)
        win_length = n_fft   # Window length

        self.anm_f_dict = dict()
        for i in range(len(anm_t_list)):
            anm_t = torch.tensor(anm_t_list[i]).to(torch.float32)
            order = torch.sqrt(torch.tensor(anm_t.shape[-1])).to(int) - 1
            Y_p = utils.create_sh_matrix(
                order, zen=self.P_th, azi=self.P_ph, type='real'
            )
            projected_values = (Y_p @ anm_t[t])
            utils.plot_on_2D(
                azi=self.P_ph,
                zen=self.P_th,
                values=projected_values,
                title=f"Encoded Signal N={order}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
            )
            # column wise STFT (channels)
            anm_f = torch.stft(anm_t.t(),#torch.sum(anm_f, dim=1).t(),
                n_fft=n_fft,
                hop_length= hop_length,
                win_length=win_length,
                window = torch.hann_window(win_length),
                return_complex=True,
                )
            plt.figure()
            plt.plot((torch.abs(anm_f[0])**2).sum(1))
            self.anm_f_dict[order] = anm_f
        self.input_order = min(self.anm_f_dict.keys())
        self.output_order = max(self.anm_f_dict.keys())
        anm_f_input_order =  self.anm_f_dict[self.input_order]
        anm_f_output_order = self.anm_f_dict[self.output_order]

        if debug:
            # project first time sample on 162 points
            for order,anm_f in zip([self.input_order,self.output_order],[anm_f_input_order,anm_f_output_order]):
                t = 10
                Y_p = utils.create_sh_matrix(
                    order, zen=self.P_th, azi=self.P_ph, type=SH_type
                )
                projected_values = (torch.abs((Y_p @ anm_f[:,:,t] ))**2).sum(dim=1)
                utils.plot_on_2D(
                    azi=self.P_ph,
                    zen=self.P_th,
                    values=projected_values,
                    title=f"Encoded Signal N={order}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
                )
        return anm_f_input_order,anm_f_output_order

    def create(
        self,
        signals: List[signal_info],
        input_order: int,
        output_order: int,
        normalize_signals: bool = False,
        SH_type: str = "complex",
        grid_type: str = LEBEDEV,
        n_fft: int = 1024,
        debug : bool = False,
        sr: Optional[int] = -1,
    ) -> None:
        self.signals, self.sr = self._align_sr(signals, force_sr=sr)

        self.has_mask = False
        self.input_order = input_order
        self.output_order = output_order

        total_anm_f_input_order = self._build_joint_soundfield(self.input_order,SH_type,n_fft,normalize_signals)
        total_anm_f_output_order = self._build_joint_soundfield(self.output_order,SH_type,n_fft,normalize_signals)

        # total_anm_t = total_anm_t / torch.sqrt(torch.sum(total_anm_t ** 2))
        self.P_th, self.P_ph, self.num_grid_points = create_grid(grid_type)

        if debug:
            # project first time sample on 162 points
            for order,anm_f in zip([self.input_order,self.output_order],[total_anm_f_input_order,total_anm_f_output_order]):
                Y_p = utils.create_sh_matrix(
                    order, zen=self.P_th, azi=self.P_ph, type=SH_type
                )
                t = 0
                projected_values = (torch.abs((Y_p @ anm_f[:,:,t] ))**2).sum(dim=1)
                # plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
                utils.plot_on_2D(
                    azi=self.P_ph,
                    zen=self.P_th,
                    values=projected_values,
                    title=f"Encoded Signal N={order}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
                )
        return total_anm_f_input_order,total_anm_f_output_order

    def __len__(self):
        if hasattr(self, "anm_t"):
            return self.anm_t.shape[0]
        else:
            print("Sound field not created yet")
            return 0

    def save_sound_field(self, dir: str) -> None:
        mask_str = "mask" if self.has_mask else "NOmask"
        timestamp = datetime.now().strftime("%yY_%mM_%dF_%HH_%Mm")
        file_name = f"sound_field_{self.input_order}_order_{self.num_windows}_win_{self.num_bins}_bin_{self.num_grid_points}_points_{mask_str}_{timestamp}.pt"
        os.makedirs(dir, exist_ok=True)
        file_name = os.path.join(dir, file_name)
        torch.save(self, file_name)
        print(f"Sound field saved to {file_name}")

    def load_from_file(self, file_path: str) -> None:
        # TODO
        pass

    def _align_sr(
        self, signals: List[signal_info], force_sr: int = -1
    ) -> List[signal_info]:
        # if force_sr is -1, the minimum sr will be used
        fs_list = torch.tensor([sig.sr for sig in signals])
        new_sr = min(fs_list) if force_sr == -1 else force_sr
        if min(fs_list) < force_sr:
            print(
                f"Warning: The minimum sr is {min(fs_list)} and the requested sr is {force_sr}. The signals will be resampled to {min(fs_list)}"
            )

        signals_to_resample = (fs_list != new_sr).nonzero()
        if len(signals_to_resample) > 0:
            for i in signals_to_resample[0]:
                assert (
                    fs_list[i] % new_sr == 0
                ), f"Error in sr for file {signals[i].name} (new sr is {new_sr} and it has {fs_list[i]})"
                signals[i].signal = utils._resample(
                    signals[i].signal, signals[i].sr, new_sr
                )
                signals[i].sr = new_sr
        return signals, new_sr

    def _create_grid(self, grid_type):
        if grid_type == LEBEDEV:
            self.num_grid_points = 2702  # P
            lebedev = scipy.io.loadmat("Lebvedev2702.mat")
            self.P_th = torch.tensor(lebedev["th"].reshape(-1))  # rad
            self.P_ph = torch.tensor(lebedev["ph"].reshape(-1))  # rad
            self.P_ph = (self.P_ph + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # wrap angles to [-pi,pi]
        elif grid_type == POINTS_162:
            self.num_grid_points = 162  # P
            points = utils.generate_sphere_points(162, plot=False)
            self.P_th = points[:, 1]
            self.P_ph = points[:, 2]
        else:
            raise ValueError(f"Unknown grid type {grid_type}")

    def get_sparse_dict_v3(
        self,
        stft_anmt: torch.tensor,
        opt: optimizer,
        mask=None,
        save=False,
        gt_stft_anmt = None,
        gt_sparse= None,
    ):
        
        opt.init(stft_anmt,gt_stft_anmt)
        self.sparse_stft_dict = opt()
        return self.sparse_stft_dict

    def plot_sparse_dict(self, s_dict, sample_idx: int):
        if s_dict.dim() == 4:
            s_dict = torch.sum(s_dict, axis=1)
            s_dict = s_dict.permute(1, 0, 2).reshape(
                self.num_grid_points, self.window_length * self.num_windows
            )
        utils.plot_on_2D(
            azi=self.P_ph,
            zen=self.P_th,
            values=s_dict[:, sample_idx].cpu(),
            title=f"Sparse Dict t={sample_idx}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
        )

    def play_sparse_stft_sound_field(
            self,
            theta: float,
            phi: float,
            sound: Optional[torch.tensor] = None,
            radius: float = 5,
            window: Union[int, None] = None,
            bin: Union[int, None] = None,
        ):
        # sound should be (num_grid_points,time samples)
        num_windows,num_grid_points,num_bins = self.sparse_stft_dict.shape
        num_bins = num_bins//2
        complex_sparse_stft_dict = torch.complex(self.sparse_stft_dict[...,:num_bins],self.sparse_stft_dict[...,num_bins:])
        mirror_part = torch.conj(complex_sparse_stft_dict[..., 1:-1].flip(dims=[-1]))
        full_spectrum = torch.cat([complex_sparse_stft_dict, mirror_part], dim=-1)
        sparse_time_dict = torch.fft.ifft(full_spectrum, dim=-1).real
        sound = sparse_time_dict.reshape(num_windows,num_grid_points*num_bins*2)

        grid_in_degress = torch.stack((self.P_ph, self.P_th)).T * 180 / torch.pi
        target_in_degress = torch.tensor([phi, theta]).reshape(1, -1)
        relevant_grid_points = (
                (torch.norm(grid_in_degress - target_in_degress, p=2, dim=1) < radius)
                .nonzero()
                .flatten()
            )

        if len(relevant_grid_points) > 0:
            print("## Playing Directions ##")
            for i in relevant_grid_points:
                print(f"Theta = {grid_in_degress[i,0]}, Phi = {grid_in_degress[i,1]}")
        else:
            print("No directions found")
            return

        signal = torch.sum(sound[relevant_grid_points, :], axis=0)
        sd.play(signal.cpu().numpy(), self.sr)

    def play_sparse_sound_field(
        self,
        theta: float,
        phi: float,
        sound: Optional[torch.tensor] = None,
        radius: float = 5,
        window: Union[int, None] = None,
        bin: Union[int, None] = None,
    ):
        # sound should be (num_grid_points,time samples)
        if sound is None:
            if window is not None:
                if bin is not None:
                    sound = self.sparse_dict_subbands[window, bin, :, :]
                else:
                    sound = torch.sum(
                        self.sparse_dict_subbands[window, :, :, :], axis=0
                    )
            else:
                if bin is not None:
                    sound = (
                        self.sparse_dict_subbands[:, bin, :, :]
                        .permute(1, 0, 2)
                        .reshape(
                            self.num_grid_points, self.window_length * self.num_windows
                        )
                    )
                else:
                    sound = (
                        torch.sum(self.sparse_dict_subbands, axis=1)
                        .permute(1, 0, 2)
                        .reshape(
                            self.num_grid_points, self.window_length * self.num_windows
                        )
                    )
        assert (
            sound.dim() == 2 and self.num_grid_points in sound.shape
        ), f"Sound wrong format {sound.shape}"
        if sound.shape[1] == self.num_grid_points:
            sound = sound.t()
        print(sound.shape)
        grid_in_degress = torch.stack((self.P_ph, self.P_th)).T * 180 / torch.pi
        target_in_degress = torch.tensor([phi, theta]).reshape(1, -1)
        relevant_grid_points = (
            (torch.norm(grid_in_degress - target_in_degress, p=2, dim=1) < radius)
            .nonzero()
            .flatten()
        )

        if len(relevant_grid_points) > 0:
            print("## Playing Directions ##")
            for i in relevant_grid_points:
                print(f"Theta = {grid_in_degress[i,0]}, Phi = {grid_in_degress[i,1]}")
        else:
            print("No directions found")
            return

        signal = torch.sum(sound[relevant_grid_points, :], axis=0)
        sd.play(signal.cpu().numpy(), self.sr)


class SoundFieldDataset(Dataset):
    def __init__(self, folder_path : str, dataset_type : str, input_order : bool = 1 , output_order : bool = 3):
        self.zip_path = os.path.join(folder_path,dataset_type + ".zip")
        self.dataset_type = dataset_type
        self.meta_data = pd.read_csv(os.path.join(folder_path,'metadata.csv')).query('type == @self.dataset_type')
        self.base_id = self.meta_data.iloc[0]['ID'] #there is an offset for each data type
        self.sr = self.meta_data.iloc[0].sr #assume all have the same SR
        self.input_order = input_order
        self.output_order = output_order

    def __len__(self):
        # Return the number of samples stored in the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zipf:
            return len([name for name in zipf.namelist() if name.startswith(self.dataset_type)])

    def __getitem__(self, local_id):
    # Lazy load each sample from the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zipf:
            with zipf.open(f"{self.dataset_type}_{self.base_id + local_id}.pth") as f:
                # self.base_id + local_id = global_id
                global_id, anm_f_input_order, anm_f_output_order = torch.load(f,weights_only=False)
        return global_id, anm_f_input_order.permute(-1,0,1), anm_f_output_order.permute(-1,0,1) #TODO just save it the correct permute when making dataset
