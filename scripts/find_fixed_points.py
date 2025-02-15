import torch.utils
import torch.utils.data
from nm.model import RNN, fpf_rnn
from nm.tasks import BehavioralTasks
from fpf.FixedPointFinderTorch import FixedPointFinderTorch
import torch
import hydra
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
from functools import partial

@hydra.main(version_base=None, config_path='../config', config_name='fpf')
def main(cfg: DictConfig):
    
    # Set up task
    wholepop_dataset = BehavioralTasks(cfg.task)
    dataloader = torch.utils.data.DataLoader(wholepop_dataset, batch_size=1, shuffle=False)

    # Load model
    model = RNN()
    save_path = Path(cfg.filepaths.models) / cfg.day / cfg.time / 'model_weights.pth'
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    for i, batch in enumerate(dataloader):
        u, nm_signal, _ = batch
        rnn_wrapper = partial(fpf_rnn, model=model, nm_signal=nm_signal)
        x0 = torch.normal(mean=0, std=1, size=(1, model.dh, 1)) / cfg.model.x0_scl
        _, x = model(u, x0, nm_signal)
        x = torch.transpose(x, dim0=1, dim1=2) # batch_size x t x dh
        x = x.detach().numpy()

        fixed_point_finder = FixedPointFinderTorch(rnn_wrapper, **cfg.fpf_hps)

        initial_states = fixed_point_finder.sample_states(
            state_traj=x,
            n_inits=cfg.n_inits,
            noise_scale=cfg.noise_scale
        )
        inputs = np.zeros(shape=(1, u.shape[-1])) # 1 x din
        unique_fps, _ = fixed_point_finder.find_fixed_points(
            initial_states,
            inputs
        )

        unique_fps.save(f"fixed_points_beh_nm_ctx_{i}.pickle")

if __name__ == '__main__':
    main()

