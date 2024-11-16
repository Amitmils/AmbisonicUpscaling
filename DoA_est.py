import torch
import matplotlib.pyplot as plt
import dataclasses
from typing import List, Union
from collections import Counter
import numpy as np


@dataclasses.dataclass
class DoA_candidate:
    zen: int
    azi: int
    count: int = 0
    band: int = -1
    window: int = -1


class DoA_via_bands:
    def __init__(self, sound_field, device) -> None:
        self.sound_field = sound_field
        self.device = device
        self.normalized_abs_sound_field = self._get_normalized_sound_field()

    def _get_normalized_sound_field(self):
        abs_sound_field = torch.abs(self.sound_field.sparse_dict_subbands)
        normalization_per_t = torch.sum(abs_sound_field, dim=2)  # or max
        normalized_abs_sound_field = abs_sound_field / normalization_per_t.unsqueeze(-2)
        return normalized_abs_sound_field


    def _get_window_candidates(self, TH=0.35):
        ids = torch.nonzero(self.normalized_abs_sound_field > TH)
        window_dir_candidates = dict()
        th_phi = (
            torch.vstack((self.sound_field.P_th, self.sound_field.P_ph)).T.cpu().numpy()
        )

        for win in range(self.sound_field.num_windows):
            window_dir_candidates[win] = torch.zeros((1, 3))
            curr_win_id = ids[torch.nonzero(ids[:, 0] == win).flatten(), 1:]
            for bin in set(curr_win_id[:, 0].cpu().numpy()):
                curr_win_bin_candidates = (
                    curr_win_id[torch.nonzero(curr_win_id[:, 0] == bin).flatten(), 1]
                    .cpu()
                    .numpy()
                )
                if len(curr_win_bin_candidates) == 0:
                    continue
                c = Counter(
                    [
                        tuple(th_phi[point_in_grid] * 180 / torch.pi)
                        for point_in_grid in curr_win_bin_candidates
                    ]
                )
                counter = c.most_common()
                window_dir_candidates[win] = [
                    DoA_candidate(
                        candidate[0][0], candidate[0][1], candidate[1], bin, win
                    )
                    for candidate in counter
                ]
        return window_dir_candidates

    def plot_window_candidates(self, window_candidates: List[DoA_candidate]):
        plt.imshow(self.normalized_abs_sound_field[0].cpu().numpy())
        plt.show()

    def get_DoA(self, TH=0.35):
        window_candidates = self._get_window_candidates()
        return window_candidates