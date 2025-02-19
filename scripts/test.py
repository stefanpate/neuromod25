import torch.utils
import torch.utils.data
from nm.model import RNN
from nm.tasks import BehavioralTasks
import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from pathlib import Path

@hydra.main(version_base=None, config_path='../config', config_name='test')
def main(cfg: DictConfig):
    wholepop_dataset = BehavioralTasks(cfg.task)
    dataloader = torch.utils.data.DataLoader(wholepop_dataset, batch_size=1, shuffle=False)

    model = RNN()
    save_path = Path(cfg.filepaths.models) / cfg.day / cfg.time / str(cfg.job_num) / 'model_weights.pth'

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    fig, ax = plt.subplots(nrows=len(wholepop_dataset), sharex=True, sharey=True)
    for i, batch in enumerate(dataloader):
        u, nm_signal, target = batch
        x0 = torch.normal(mean=0, std=1, size=(1, model.dh, 1)) / cfg.model.x0_scl
        output, x = model(u, x0, nm_signal)

        target = target.squeeze().detach().numpy()
        output = output.squeeze().detach().numpy()

        ax[i].plot(target, 'k--', label="Target")
        ax[i].plot(output, 'r-', label="Output")

    fig.savefig("test.png")

if __name__ == '__main__':
    main()