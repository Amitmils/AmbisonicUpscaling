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


LEBEDEV_GRID_PATH = "Lebvedev2702.mat"
LEBEDEV = "lebedev"
POINTS_162 = "162_points"


def create_grid(grid_type):
    if grid_type == LEBEDEV:
        num_grid_points = 2702  # P
        lebedev = scipy.io.loadmat("Lebvedev2702.mat")
        P_th = torch.tensor(lebedev["th"].reshape(-1))  # rad
        P_ph = torch.tensor(lebedev["ph"].reshape(-1))  # rad
        P_ph = (P_ph + torch.pi) % (
            2 * torch.pi
        ) - torch.pi  # wrap angles to [-pi,pi]
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
    assert downsample == 1, "Downsample is not supported yet"
    # self.sr = self.sr / downsample #FIX - if downsample > 1 we need a LPF

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
            (num_bins + 2, num_samples, num_coeff),device=anm_t.device
        )  # num_bins + low and high for perfect reconstruction  | filter_length = num of SH coeff | num_samples = t
        for coeff in range(num_coeff):
            erb_bank.generate_subbands(anm_t[:, coeff])
            anm_t_subbands[:, :, coeff] = torch.tensor(erb_bank.subbands.T).clone().detach()

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
    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device

    def create(
        self,
        signals: List[signal_info],
        order: int,
        normalize_signals: bool = False,
        SH_type: str = "real",
        grid_type: str = LEBEDEV,
        debug=False,
        sr: Optional[int] = -1,
    ) -> None:
        signals, self.sr = self._align_sr(signals, force_sr=sr)
        self.anm_t_list = []
        self.y_list = []
        self.sources_coords = []
        self.has_mask = False
        self.input_order = order
        max_length = 0

        for curr_sig in signals:
            self.sources_coords.append(
                (math.degrees(curr_sig.th), math.degrees(curr_sig.ph))
            )
            anm_t, _, _, y = utils.encode_signal(
                curr_sig,
                order,
                plot=False,
                type=SH_type,
                normalize_signal=normalize_signals,
            )
            max_length = max(max_length, anm_t.shape[0])
            self.anm_t_list.append(anm_t)
            self.y_list.append(y)

        # combine all signals
        total_anm_t = torch.zeros((max_length, self.anm_t_list[0].shape[1]))
        for i in range(len(self.anm_t_list)):
            total_anm_t += torch.nn.functional.pad(
                self.anm_t_list[i],
                (0, 0, 0, max_length - self.anm_t_list[i].shape[0]),
            )

        self.P_th, self.P_ph, self.num_grid_points = create_grid(grid_type)
        Y_p = utils.create_sh_matrix(order, zen=self.P_th, azi=self.P_ph, type=SH_type)

        if debug:
            # project first time sample on 192 points
            t = 0
            projected_values = total_anm_t[t, :] @ torch.conj(Y_p)
            # plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
            utils.plot_on_2D(
                azi=self.P_ph,
                zen=self.P_th,
                values=projected_values,
                title=f"Encoded Signal N={order}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
            )
        return total_anm_t

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
        if min(fs_list)<force_sr:
            print(f"Warning: The minimum sr is {min(fs_list)} and the requested sr is {force_sr}. The signals will be resampled to {min(fs_list)}")
        
        signals_to_resample = (fs_list != new_sr).nonzero()
        if len(signals_to_resample) > 0:
            for i in signals_to_resample[0]:
                assert (
                    fs_list[i] % new_sr == 0
                ), f"Error in sr for file {signals[i].name} (new sr is {new_sr} and it has {fs_list[i]})"
                signals[i].signal = utils._resample(
                    signals[i].signal, signals[i].sr, new_sr
                )
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

    def get_sparse_dict_v2(
        self,
        windowed_anm_t : torch.tensor,
        opt: optimizer,
        mask=None,
        iter=1e5,
        multi_processing: bool = True,
        save=False,
    ):
        Bk_matrix = windowed_anm_t.permute(
            0, 1, 3, 2
        )  # turn to (window,band,SH_coeff,time)
        if mask is not None:
            mask_matrix = mask[None, None, ...]
        else:
            mask_matrix = None
        self.sparse_dict_subbands, Dk = opt.optimize(
            Bk_matrix, itr=iter, mask=mask_matrix, D_prior=None
        )
        self.s_windowed = torch.sum(self.sparse_dict_subbands, axis=1)
        self.s_dict = self.s_windowed.permute(1, 0, 2).reshape(
            self.num_grid_points, self.window_length * self.num_windows
        )
        if save:
            self.save_sound_field("data/output")
        return self.sparse_dict_subbands, self.s_windowed, self.s_dict

    def get_sparse_dict(self, opt: optimizer, mask=None, multi_processing: bool = True):
        spare_dict_subbands = torch.zeros(
            (self.num_windows, self.num_bins, self.num_grid_points, self.window_length)
        )
        if multi_processing:
            print("Multi Processing")
            args = [
                (self.windowed_anm_t[window, band, :, :].T[None, None, ...], mask, None)
                for window in range(self.num_windows)
                for band in range(self.num_bins)
            ]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = []
                with tqdm(total=len(args)) as pbar:
                    for result in pool.imap(opt.optimize, args):
                        results.append(result)
                        pbar.update()
                for i, (s_subband, Dk) in enumerate(results):
                    window = i // self.num_bins
                    band = i % self.num_bins
                    spare_dict_subbands[window, band, :, :] = s_subband
        else:
            # Create a single progress bar for the outer loop (bands)
            outer_bar = tqdm(total=self.num_bins, desc="Bands", position=0, leave=True)
            for band in range(self.num_bins):
                outer_bar.set_postfix(
                    {"Current Band": band}
                )  # Update current band in the outer bar
                inner_bar = tqdm(
                    total=self.num_windows, desc="Windows", position=1, leave=False
                )  # Inner bar for windows
                for window in range(self.num_windows):
                    Bk = self.windowed_anm_t[window, band, :, :].T
                    spare_dict_subbands[window, band, :, :], Dk = opt.optimize(
                        Bk, mask, D_prior=None
                    )
                    inner_bar.update(1)  # Update inner progress bar
                inner_bar.close()  # Close the inner progress bar after finishing the inner loop
                outer_bar.update(1)  # Update outer progress bar

        self.sparse_dict_subbands = spare_dict_subbands
        self.s_windowed = torch.sum(self.sparse_dict_subbands, axis=1)
        self.s_dict = self.s_windowed.permute(1, 0, 2).reshape(
            self.num_grid_points, self.window_length * self.num_windows
        )

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
            title=f"Encoded Signal N={self.input_order} t={sample_idx}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
        )

    def play_sparse_sound_field(
        self,
        theta: float,
        phi: float,
        radius: float = 5,
        window: Union[int, None] = None,
        bin: Union[int, None] = None,
    ):
        # sound should be (num_grid_points,time samples)
        if window is not None:
            if bin is not None:
                sound = self.sparse_dict_subbands[window, bin, :, :]
            else:
                sound = torch.sum(self.sparse_dict_subbands[window, :, :, :], axis=0)
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
    def __init__(self, data: dict,device : str = 'cpu') -> None:
        super().__init__()
        self.data = data["data"]
        self.device = device
        for key, value in data["config"].items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0].to(self.device),self.data[idx][1].to(self.device)

