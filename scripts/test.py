import torch.utils
import torch.utils.data
from nm.model import RNN
from nm.tasks import BehavioralTasks
import torch
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='../config', config_name='test')
def main(cfg: DictConfig):
    wholepop_dataset = BehavioralTasks(cfg.task)
    dataloader = torch.utils.data.DataLoader(wholepop_dataset, batch_size=1, shuffle=True)

if __name__ == '__main__':
    main()