import torch
from torch import nn
from nm.model import RNN
from nm.tasks import BehavioralTasks
import hydra
from omegaconf import DictConfig

def train_loop(dataloader, model, loss_fn, optimizer, x0_scl, batch_size):
    model.train()
    for u, nm_signal, target in dataloader:
        # Init states
        x0 = torch.normal(mean=0, std=1, size=(batch_size, model.dh, 1)) / x0_scl
        r0 = torch.sigmoid(x0)

        # Compute prediction and loss
        output, x = model(u, r0, nm_signal)
        loss = loss_fn(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss={loss}")

@hydra.main(version_base=None, config_path='../config', config_name='train')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    # Task / data
    wholepop_dataset = BehavioralTasks(cfg.task)
    dataloader = torch.utils.data.DataLoader(wholepop_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Model
    model = RNN()

    # Training
    mse = nn.MSELoss()
    loss_fn = lambda pred, target : torch.sqrt(mse(pred, target))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Train
    for t in range(cfg.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            x0_scl=cfg.model.x0_scl,
            batch_size=cfg.batch_size,
        )
    
    print("Done!")
    

if __name__ == "__main__":
    main()