import torch
from torch import nn
import numpy as np

class RNN(nn.Module):
    def __init__(
            self,
            seed: int = 1234,
            pinh: float=0.2,
            pcon: float = 0.8,
            g: float = 1.5,
            di: int = 1,
            dh: int =200,
            do: int = 1,
            dt: int = 1,
            tau_range: tuple = (4, 20), 
            wout_scl: int = 100,
            x_noise_scl: int = 10
        ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.x_noise_scl = x_noise_scl
        
        # Non-trainable parameters
        self.win = torch.nn.Parameter(
            torch.normal(mean=0, std=1, size=(dh, di)),
            requires_grad=False
        )

        self.m = torch.nn.Parameter(
            torch.eye(dh),
            requires_grad=False
        )
        inh_idxs = self.rng.choice(dh, size=int(dh * pinh), replace=False)
        self.m[inh_idxs, inh_idxs] = -1 # Dale's law

        # Trainable parameters
        self.wout = torch.nn.Parameter(
            torch.normal(mean=0, std=1, size=(do, dh)) / wout_scl,
            requires_grad=True,
        )

        self._w = torch.nn.Parameter(
            torch.normal(mean=0, std= g / np.sqrt(dh * pcon), size=(dh, dh)),
            requires_grad=True
        )
        
        self.taus = torch.nn.Parameter(
            torch.sigmoid(
                torch.normal(mean=0, std=1, size=(dh, 1)),
            ) * (tau_range[1] - tau_range[0]) + tau_range[0]
        )


    def single_step(self, u: torch.Tensor, x: torch.Tensor):
        r = torch.sigmoid(x)
        decay_term = torch.multiply(1 - self.dt / self.taus, x)
        recurrent_term = torch.multiply(
            self.dt / self.taus,
            torch.matmul(self.w, r)
        )
        input_term = torch.matmul(self.win, u)
        noise = torch.normal(mean=0, std=1, size=x.shape) / self.x_noise_scl
        x = decay_term + recurrent_term + input_term + noise
        output = torch.matmul(self.wout, torch.sigmoid(x)) + self.bias

        return output, x

    # Call this rnn for now to work with fpf
    def rnn(self, u: torch.Tensor, r0: torch.Tensor):
        T = u.shape[0]
        outputs = []
        xs = []
        for t in range(T):
            if t == 0:
                r = r0
            else:
                r = torch.sigmoid(x)
            
            output, x = self.single_step(u[t], r0)
            outputs.append(output)
            xs.append(x)

        return torch.stack(outputs), torch.stack(xs)

    def forward(self, u: torch.Tensor, r0: torch.Tensor):
        return self.rnn(u, r0)
    

    @property
    def w(self):
        return torch.abs(torch.matmul(self._w, self.m))

if __name__ == '__main__':
    model = RNN()
    print(model.state_dict())