import torch
import matplotlib.pyplot as plt
import dataclasses
from typing import List, Union
from collections import Counter
import numpy as np
import utils
import gc


@dataclasses.dataclass(frozen=True, eq=True)
class DoA_candidate:
    zen: int
    azi: int
    count: int = 0
    band: int = -1
    window: int = -1
    
    def __eq__(self, other: object) -> bool:
        if (self.zen == other.zen) and (self.azi == other.azi):
            return True
        else:
            return False

    def __hash__(self) -> int:
        # Custom hash: Only hash based on zen and azi
        return hash((self.zen, self.azi))


class DoA_via_bands:
    def __init__(self, sound_field, device , window=None) -> None:
        self.sound_field = sound_field
        self.device = device
        self.normalized_abs_sound_field = self._get_normalized_sound_field(window)

    def _get_normalized_sound_field(self, window=None):
        # Move the tensor to GPU for computation
        result_list = []
        for i in range(self.sound_field.num_windows): 
                print(f"{i+1}/{self.sound_field.num_windows}")
                # Extract the batch
                batch = torch.abs(self.sound_field.sparse_dict_subbands)[i].to(self.device)  # Shape: (2702, 1024)
                
                # Compute the normalization and normalized result for this batch
                norms = torch.sum(batch, dim=-2)
                normalized_batch = batch / norms.unsqueeze(-2)  # Broadcasting division
                
                # Move the result back to CPU
                result_list.append(normalized_batch.cpu())  # Append to the result list

        # After all batches are processed, concatenate them
        result = torch.stack(result_list, dim=0)  # Final shape will be (N, M, 2702, 1024)
        
        
        return result


    def _get_window_candidates_v2(self, TH=0.35):
        th_phi = (
            torch.vstack((self.sound_field.P_th, self.sound_field.P_ph)).T.cpu().numpy()
        )
        max_values, max_indices = torch.max(self.normalized_abs_sound_field, dim=-2)
        window_dir_candidates = dict()
        for win in range(self.sound_field.num_windows):
            print(win)
            candidates_in_win = max_indices[win, max_values[win] > TH] #in bins
            candidates_in_win = th_phi[candidates_in_win]* 180 / torch.pi #in degrees
            counter = Counter([tuple(candidate) for candidate in candidates_in_win]).most_common()
            window_dir_candidates[win] = [DoA_candidate(candidate[0][0], candidate[0][1], candidate[1], -1, win) for candidate in counter]
        return window_dir_candidates

    def _get_window_candidates(self, TH=0.35):
        ids = torch.nonzero(self.normalized_abs_sound_field > TH)
        window_dir_candidates = dict()
        th_phi = (
            torch.vstack((self.sound_field.P_th, self.sound_field.P_ph)).T.cpu().numpy()
        )

        for win in range(self.sound_field.num_windows):
            window_dir_candidates[win] = list()
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
                window_dir_candidates[win].extend(
                    [
                        DoA_candidate(
                            candidate[0][0], candidate[0][1], candidate[1], bin, win
                        )
                        for candidate in counter
                    ]
                )
            window_dir_candidates[win] = Counter(window_dir_candidates[win]).most_common()
        return window_dir_candidates

    def plot_window_candidates(self, window_candidates: List[DoA_candidate],window : int = None):
        def plot():
            x,y,c = list(),list(),list()
            for candidate in window_candidates[window]:
                x.append(candidate.zen)
                y.append(candidate.azi)
                c.append(candidate.count)
            plt.figure()
            plt.title(f"DoA candidates for window {window}")
            plt.xlim(-180,180)
            plt.ylim(0,180)
            plt.scatter(x,y,c=c,cmap='viridis')
            plt.colorbar()
        def plot_v2():
            x,y,c = list(),list(),list()
            for candidate in window_candidates[window]:
                x.append(candidate.zen * torch.pi/180)
                y.append(candidate.azi * torch.pi/180)
                c.append(1)
            utils.plot_on_2D(azi=torch.tensor(y),zen=torch.tensor(x),values=torch.tensor(c),title=f"DoA candidates for window {window}")

        if window is not None:
            plot_v2()
        else:
            for window in range(len(window_candidates)):
                plot_v2()


    def get_DoA(self, TH=0.35):
        window_candidates = self._get_window_candidates()
        return window_candidates
