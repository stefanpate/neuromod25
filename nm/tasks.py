import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
import hydra
import numpy as np

class BehavioralTasks(Dataset):
    def __init__(self, task: DictConfig):
        super().__init__()
        self.task = task
        self.rng = np.random.default_rng(seed=self.task.seed)
        self.neuron_idxs = np.arange(self.task.dh)
        self.rng.shuffle(self.neuron_idxs)

    def __len__(self):
        return len(self.task.contexts)
    
    def __getitem__(self, idx: int):
        u = torch.zeros(size=(self.task.T, 1))
        target = torch.zeros(size=(self.task.T, 1))
        nm_signal = torch.eye(n=self.task.dh)

        ctx = self.task.contexts[idx]
        n_to_scl = int(self.task.dh * (self.task.scl_percent / 100))
        
        u[self.task.stim_on : self.task.stim_off] = ctx['stim']
        target[self.task.target_on : self.task.target_off] = ctx['target']

        if ctx['nm']:
            nm_signal[self.neuron_idxs[:n_to_scl], self.neuron_idxs[:n_to_scl]] *= self.task.nm_scl

        return u, nm_signal, target


@hydra.main(version_base=None, config_path="../config/task", config_name="wholepop")
def main(cfg: DictConfig):
    btasks = BehavioralTasks(cfg)
    u, nm, target = btasks.__getitem__(2)
    print()

if __name__ == '__main__':
    main()
