import dataclasses
import torchaudio
import math
import torch

@dataclasses.dataclass
class signal_info():
    name : str
    signal_path : str 
    th : float = 0
    ph : float = 0
    in_rad : bool = False
    device : torch.device = torch.device("cpu")

    def __post_init__(self):
        self.th = self.th if self.in_rad else math.radians(self.th)
        self.ph = self.ph if self.in_rad else math.radians(self.ph)
        self.signal,self.sr = torchaudio.load(self.signal_path)
        self.signal = self.signal.to(self.device)

    def __len__(self):
        return len(self.signal)