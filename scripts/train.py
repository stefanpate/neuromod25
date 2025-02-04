import torch
from torch import nn
from nm.model import RNN
from nm.tasks import BehavioralTasks
import hydra
from omegaconf import DictConfig
from collections import deque
import numpy as np

def train_loop(dataloader, model, loss_fn, optimizer, x0_scl, batch_size):
    model.train()
    losses = []
    for u, nm_signal, target in dataloader:
        # Init state
        x0 = torch.normal(mean=0, std=1, size=(batch_size, model.dh, 1)) / x0_scl

        # Compute prediction and loss
        output, x = model(u, x0, nm_signal)
        loss = loss_fn(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss={loss / batch_size}")
        losses.append(float(loss.detach().numpy()))

    return losses

@hydra.main(version_base=None, config_path='../config', config_name='train')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    # Task / data
    wholepop_dataset = BehavioralTasks(cfg.task)
    dataloader = torch.utils.data.DataLoader(wholepop_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Model
    model = RNN()

    # Training
    mse = nn.MSELoss(reduction='sum')
    loss_fn = lambda pred, target : torch.sqrt(mse(pred, target))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    
    # Train
    last_x_loss = deque(maxlen=int((cfg.last_x_scl * cfg.task.n_bctx) / cfg.batch_size))
    epoch_losses = []    
    
    for t in range(cfg.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = train_loop(
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            x0_scl=cfg.model.x0_scl,
            batch_size=cfg.batch_size,
        )

        epoch_losses.append(sum(losses) / (len(losses) * cfg.batch_size))
        last_x_loss.extend(losses)

        if sum(last_x_loss) / (len(last_x_loss) * cfg.batch_size) < cfg.perf_threshold:
            print("Met performance criteria")
            break
    
    print("Saving")
    epoch_losses = np.array(epoch_losses)
    np.save("epoch_losses", epoch_losses)
    torch.save(model.state_dict(), 'model_weights.pth')
    

if __name__ == "__main__":
    main()