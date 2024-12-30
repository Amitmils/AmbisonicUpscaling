import torch
from optimizer_v2 import optimizer_v2
from datetime import datetime
import os
from typing import List, Union
from sound_field import SoundField


class unfolded_optimizer:
    def __init__(
        self,
        sound_field: SoundField,
        Y_p: torch.tensor,
        alpha: int,
        num_iters: int,
        device: str = "cpu",
        deep_unfolded: bool = True,
        load_model: Union[str, None] = None,
    ) -> None:

        today = datetime.today()
        now = datetime.now()
        self.run_name = f"{today.strftime('D%d_M%m')}_{now.strftime('h%H_m%M')}"
        self.run_folder = os.path.join("runs", self.run_name)
        os.makedirs(self.run_folder, exist_ok=True)
        if load_model is not None:
            self.unfolded_optimizer = torch.load(load_model)
        else:
            self.unfolded_optimizer = optimizer_v2(
                sound_field, Y_p, alpha, num_iters, device, deep_unfolded
            )
        self.optimizer = torch.optim.Adam(self.self.unfolded_optimizer.parameters(), lr=1e-3)

    def train(self, num_epochs: int):
        self.unfolded_optimizer.train()
        best_loss = torch.inf
        for epoch in range(num_epochs):
            #TODO dataloader
            upscaled_anm = self.unfolded_optimizer()
            loss = 0
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if loss < best_loss:
                best_loss = loss
                torch.save(self.unfolded_optimizer, os.path.join(self.run_folder, "best_model.pth"))
            print(f"Epoch: {epoch}, Loss: {loss}")