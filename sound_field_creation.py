import utils
import math
import dataclasses
import torchaudio
import multiprocessing as mp
from typing import List, Union
import scipy.io
from py_bank.filterbanks import EqualRectangularBandwidth
from signal_info import signal_info
from optimizer import optimizer
from tqdm import tqdm
import torch

LEBEDEV_GRID_PATH = "Lebvedev2702.mat"
LEBEDEV = "lebedev"
POINTS_162 = "162_points"


class sound_field:
    def __init__(self,device: torch.device = torch.device("cpu"),
) -> None:
        self.device = device

    def create(
        self,
        signals: List[signal_info],
        order: int,
        normalize_signals: bool = False,
        SH_type: str = "real",
        grid_type: str = LEBEDEV,
        debug=False,
    ) -> None:
        signals, self.sr = self._ensure_same_sr(signals)
        self.anm_t_list = []
        self.y_list = []
        self.sources_coords = []
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
        total_anm_t = torch.zeros((max_length, self.anm_t_list[0].shape[1])).to(self.device)
        for i in range(len(self.anm_t_list)):
            total_anm_t += torch.nn.functional.pad(
                self.anm_t_list[i],
                (0, 0, 0, max_length - self.anm_t_list[i].shape[0]),
            )

        self._create_grid(grid_type)
        Y_p = utils.create_sh_matrix(order, zen=self.P_th, azi=self.P_ph, type=SH_type).to(self.device)
        
        if debug:
            # project first time sample on 192 points
            t = 0
            projected_values = total_anm_t[t, :] @ torch.conj(Y_p)
            # plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
            utils.plot_on_2D(
                azi=self.P_ph,
                zen=self.P_th,
                values=projected_values,
                title=f"Encoded Signal N={order}\n$\\theta$ = {math.degrees(curr_sig.th)} $\\phi$ = {math.degrees(curr_sig.ph)}"
            )

        self.anm_t = total_anm_t
        return self.anm_t

    def load_from_file(self, file_path: str) -> None:
        # TODO
        pass

    def divide_to_subbands(
        self, num_bins: int, downsample: int = 2, low_filter_center_freq: int = 1
    ) -> torch.tensor:
        # signal is size [num_samples,(ambi Order+1)^2]
        assert hasattr(self, "anm_t"), "You must first create the sound field"
        anm_t = self.anm_t[::downsample]
        self.sr = self.sr / downsample
        self.num_bins = num_bins

        if num_bins == 1:
            self.anm_t_subbands = anm_t[None, ...]
        else:
            num_bins -= 2  # for perfect reconstruction (first bin is low pass and last bin is high pass)
            high_filter_center_freq = self.sr / 2  # centre freq. of highest filter
            num_samples, num_coeff = anm_t.shape  # filter bank length
            erb_bank = EqualRectangularBandwidth(
                num_samples,
                self.sr.cpu().numpy(),
                num_bins,
                low_filter_center_freq,
                high_filter_center_freq.cpu().numpy(),
            )
            anm_t_subbands = torch.zeros(
                (num_bins + 2, num_samples, num_coeff)
            )  # num_bins + low and high for perfect reconstruction  | filter_length = num of SH coeff | num_samples = t
            for coeff in range(num_coeff):
                erb_bank.generate_subbands(anm_t[:, coeff].cpu().numpy())
                anm_t_subbands[:, :, coeff] = torch.tensor(erb_bank.subbands.T)

            # [pass band k,t,SH_coeff]
            self.anm_t_subbands = anm_t_subbands.to(self.device)
        return self.anm_t_subbands

    def divide_to_time_windows(
        self, window_length: int, max_num_windows: int
    ) -> torch.tensor:
        assert hasattr(self, "anm_t_subbands"), "You must first divide to subbands"
        # signal is size [band pass k ,time samples,(ambi Order+1)^2]
        num_samples = self.anm_t_subbands.shape[1]

        # Pad the tensor
        padding = (0, 0, 0, window_length - num_samples % window_length)
        anm_t_padded = torch.nn.functional.pad(self.anm_t_subbands, padding, mode="constant", value=0)
    
        # Split the tensor into windows (Window,Freq Bin,SH Coeff,Time)
        windowed_anm_t = anm_t_padded.unfold(dimension=1, step = window_length, size = window_length).permute(1,0,3,2)
        
        self.windowed_anm_t = windowed_anm_t[:max_num_windows]
        self.num_windows = self.windowed_anm_t.shape[0]
        self.window_length = window_length
        return self.windowed_anm_t

    def _ensure_same_sr(self, signals: List[signal_info]) -> List[signal_info]:
        fs_list = torch.tensor([sig.sr for sig in signals])
        smallest_fs = min(fs_list)
        for i in (fs_list != smallest_fs).nonzero()[0]:
            assert (
                fs_list[i] % smallest_fs == 0
            ), f"Error in fs for file {signals[i].name} (min is {smallest_fs} and it has {fs_list[i]})"
            signals[i].signal = utils._resample(
                signals[i].signal, signals[i].sr, smallest_fs
            )
        return signals, smallest_fs

    def _create_grid(self, grid_type):
        if grid_type == LEBEDEV:
            self.num_grid_points = 2702  # P
            lebedev = scipy.io.loadmat("Lebvedev2702.mat")
            self.P_th = lebedev["th"].reshape(-1)  # rad
            self.P_ph = lebedev["ph"].reshape(-1)  # rad
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
        self.P_ph = torch.tensor(self.P_ph)
        self.P_th = torch.tensor(self.P_th)

    def get_sparse_dict_v2(self, opt: optimizer, mask=None, multi_processing: bool = True):
        if hasattr(self, "sparse_dict_subbands"):
            try:
                del self.sparse_dict_subbands
            except:
                pass
            try:
                del self.s_windowed
            except:
                pass
            try:
                del self.s_dict
            except:
                pass
        Bk_matrix = self.windowed_anm_t.permute(0,1,3,2) #turn to (window,band,SH_coeff,time)
        if mask is not None:
            mask_matrix = mask[None,None,...]
        else:
            mask_matrix = None
        self.sparse_dict_subbands,Dk = opt.optimize(Bk_matrix, mask_matrix, None)
        self.s_windowed = torch.sum(torch.tensor(self.sparse_dict_subbands), axis=1)
        self.s_dict = self.s_windowed.permute(1,0,2).reshape(self.num_grid_points, self.window_length * self.num_windows)

    def get_sparse_dict(self, opt: optimizer, mask=None, multi_processing: bool = True):
        spare_dict_subbands = torch.zeros(
            (self.num_windows, self.num_bins, self.num_grid_points, self.window_length)
        )
        if multi_processing:
            print("Multi Processing")
            args = [
                (self.windowed_anm_t[window, band, :, :].T[None,None,...], mask, None)
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
        self.s_dict = self.s_windowed.permute(1,0,2).reshape(self.num_grid_points, self.window_length * self.num_windows)

    def plot_sparse_dict(self,sample_idx: int):
        utils.plot_on_2D(
            azi=self.P_ph,
            zen=self.P_th,
            values=self.s_dict[:, sample_idx].cpu(),
            title=f"Encoded Signal N={self.input_order} t={sample_idx}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
        )