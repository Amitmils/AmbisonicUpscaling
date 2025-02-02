import utils
import math
import numpy as np
import dataclasses
import torchaudio
import multiprocessing as mp
from typing import List, Union
import scipy.io
from py_bank.filterbanks import EqualRectangularBandwidth
from signal_info import signal_info
from optimizer import optimizer
from tqdm import tqdm

LEBEDEV_GRID_PATH = "Lebvedev2702.mat"
LEBEDEV = "lebedev"
POINTS_162 = "162_points"


class sound_field:
    def __init__(self) -> None:
        pass

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
        self.anm_t_list = list()
        self.y_list = list()
        self.sources_coords = list()
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
        total_anm_t = 0
        for i in range(len(self.anm_t_list)):
            total_anm_t += np.pad(
                self.anm_t_list[i],
                ((0, max_length - self.anm_t_list[i].shape[0]), (0, 0)),
            )

        self._create_grid(grid_type)
        Y_p = utils.create_sh_matrix(order, zen=self.P_th, azi=self.P_ph, type=SH_type)
        if debug:
            # project first time sample on 192 points
            t = 0
            projected_values = total_anm_t[t, :] @ np.conj(Y_p)
            # plot_on_sphere([P_th,P_ph],projected_values,title=f"Encoded Signal N={sh_order_input}\n$\\theta$ = {math.degrees(th)} $\\phi$ = {math.degrees(ph)}")
            utils.plot_on_2D(
                azi=self.P_ph,
                zen=self.P_th,
                values=projected_values,
                title=f"Encoded Signal N={order} t={t}\n$(\\theta,\\phi)$ := {[tuple((round(th),round(phi))) for (th,phi) in self.sources_coords]}",
            )

        self.anm_t = total_anm_t
        return self.anm_t

    def load_from_file(self, file_path: str) -> None:
        # TODO
        pass

    def divide_to_subbands(
        self, num_bins: int, downsample: int = 2, low_filter_center_freq: int = 1
    ) -> np.ndarray:
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
                self.sr,
                num_bins,
                low_filter_center_freq,
                high_filter_center_freq,
            )
            anm_t_subbands = np.zeros(
                (num_bins + 2, num_samples, num_coeff)
            )  # num_bins + low and high for perfect reconstruction  | filter_length = num of SH coeff | num_samples = t
            for coeff in range(num_coeff):
                erb_bank.generate_subbands(anm_t[:, coeff])
                anm_t_subbands[:, :, coeff] = erb_bank.subbands.T

            # [pass band k,t,SH_coeff]
            self.anm_t_subbands = anm_t_subbands
        return self.anm_t_subbands

    def divide_to_time_windows(
        self, window_length: int, max_num_windows: int
    ) -> np.ndarray:
        assert hasattr(self, "anm_t_subbands"), "You must first divide to subbands"
        # signal is size [band pass k ,time samples,(ambi Order+1)^2]
        num_samples = self.anm_t_subbands.shape[1]
        anm_t_padded = np.pad(
            self.anm_t_subbands,
            ((0, 0), (0, window_length - num_samples % window_length), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        windowed_anm_t = np.array_split(
            anm_t_padded, anm_t_padded.shape[1] // window_length, axis=1
        )
        windowed_anm_t = np.stack(windowed_anm_t)
        self.windowed_anm_t = windowed_anm_t[:max_num_windows]
        self.num_windows = self.windowed_anm_t.shape[0]
        self.window_length = window_length
        return self.windowed_anm_t

    def _ensure_same_sr(self, signals: List[signal_info]) -> List[signal_info]:
        fs_list = np.array([sig.sr for sig in signals])
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
            self.P_ph = (self.P_ph + np.pi) % (
                2 * np.pi
            ) - np.pi  # wrap angles to [-pi,pi]
        elif grid_type == POINTS_162:
            self.num_grid_points = 162  # P
            points = utils.generate_sphere_points(162, plot=False)
            self.P_th = points[:, 1]
            self.P_ph = points[:, 2]
        else:
            raise ValueError(f"Unknown grid type {grid_type}")

    def get_sparse_dict(self, opt: optimizer, mask=None, multi_processing: bool = True):
        spare_dict_subbands = np.zeros(
            (self.num_windows, self.num_bins, self.num_grid_points, self.window_length)
        )
        loss = np.zeros(
            (self.num_windows, self.num_bins)
        )
        if multi_processing:
            print("Multi Processing")
            args = [
                (self.windowed_anm_t[window, band, :, :].T, mask, None)
                for window in range(self.num_windows)
                for band in range(self.num_bins)
            ]
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = []
                with tqdm(total=len(args)) as pbar:
                    for result in pool.imap(opt.optimize, args):
                        results.append(result)
                        pbar.update()
                for i, (s_subband, Dk,loss) in enumerate(results):
                    window = i // self.num_bins
                    band = i % self.num_bins
                    spare_dict_subbands[window, band, :, :] = s_subband
                    # self.loss[window,band] = loss
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
                    spare_dict_subbands[window, band, :, :], Dk, loss = opt.optimize(
                        Bk, mask, D_prior=None
                    )
                    inner_bar.update(1)  # Update inner progress bar
                inner_bar.close()  # Close the inner progress bar after finishing the inner loop
                outer_bar.update(1)  # Update outer progress bar
            self.loss = loss
        return spare_dict_subbands
