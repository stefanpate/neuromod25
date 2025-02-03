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
        self.dh = dh
        
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

        self.bias = torch.nn.Parameter(
            torch.zeros(size=(do, do)),
            requires_grad=True
        )


    def single_step(self, u: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
        r = torch.sigmoid(x)
        decay_term = torch.multiply(1 - self.dt / self.taus, x)
        recurrent_term = torch.multiply(
            self.dt / self.taus,
            torch.matmul(w, r)
        )
        input_term = torch.matmul(self.win, u)
        noise = torch.normal(mean=0, std=1, size=x.shape) / self.x_noise_scl
        x = decay_term + recurrent_term + input_term + noise
        output = torch.matmul(self.wout, torch.sigmoid(x)) + self.bias
        output = torch.transpose(output, dim0=1, dim1=2) # (batch x time x do)

        return output, x

    def forward(self, u: torch.Tensor, r0: torch.Tensor, nm_signal: torch.Tensor = 1):
        T = u.shape[1]
        outputs = []
        xs = []
        
        if isinstance(nm_signal, int):
            nm_signal = torch.eye(self.dh).unsqueeze(0)
        
        w = self.get_w(nm_signal)
        for t in range(T):
            if t == 0:
                r = r0
            else:
                r = torch.sigmoid(x)
            
            u_t = u[:, t, :].unsqueeze(1)
            output, x = self.single_step(u_t, r, w)
            outputs.append(output)
            xs.append(x)

        return torch.cat(outputs, dim=1), torch.cat(xs, dim=-1)    

    def get_w(self, nm_signal: torch.Tensor):
        w_unscl = torch.abs(torch.matmul(self._w, self.m))
        return torch.matmul(w_unscl, nm_signal)

if __name__ == '__main__':
    model = RNN()
    print(model.state_dict())