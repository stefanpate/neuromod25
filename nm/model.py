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
        self.tau_range = tau_range
        
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
            torch.multiply(
                torch.rand(dh, dh) < pcon,
                torch.normal(mean=0, std=1, size=(dh, dh)) * (g / np.sqrt(dh * pcon)),
            ),
            requires_grad=True
        )
        
        self._taus = torch.nn.Parameter(
            torch.normal(mean=0, std=1, size=(dh, 1)),
            requires_grad=True
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

    def forward(self, u: torch.Tensor, x0: torch.Tensor, nm_signal: torch.Tensor = 1):
        '''
        Args
        ----
        u:Tensor
            (batch_size x T x din)
        x0:Tensor
            (batch_size x dh x 1)
        nm_sig:Tensor
            (batch_size x dh x dh)

        Returns
        -------
        outputs
            (batch_size x T x do)
        xs
            (batch_size x dh x T)
        '''
        T = u.shape[1]
        outputs = []
        xs = []
        
        if isinstance(nm_signal, int):
            nm_signal = torch.eye(self.dh).unsqueeze(0)
        
        x = x0
        w = self.get_w(nm_signal)
        for t in range(T):
            u_t = u[:, t, :].unsqueeze(1)
            output, x = self.single_step(u_t, x, w)
            outputs.append(output)
            xs.append(x)

        return torch.cat(outputs, dim=1), torch.cat(xs, dim=-1)
    

    def get_w(self, nm_signal: torch.Tensor):
        w_unscl = torch.matmul(torch.abs(self._w), self.m)
        return torch.matmul(w_unscl, nm_signal)
    
    @property
    def taus(self):
        taus = torch.sigmoid(
            self._taus
        ) * (self.tau_range[1] - self.tau_range[0]) + self.tau_range[0]
        return taus
    
def fpf_rnn(u: torch.Tensor, x0: torch.Tensor, model: RNN, nm_signal: torch.Tensor):
    '''
    Wrapper function to map FPF arguments to ours

    Args
    ----
    u:Tensor
        (batch_size x T x din)
    x0:Tensor
        (1 x batch_size x dh)

    Returns
    --------
    xs:Tensor
        (batch_size x t x dh) hidden states over time
    xT:
        (1 x batch_size x dh) final hidden state
    '''
    x0 = torch.permute(x0, (1, 2, 0))
    _, xs = model(u, x0, nm_signal)

    xs = torch.permute(xs, (0, 2, 1))
    xT = xs[:, -1, :].unsqueeze(0)

    return xs, xT


if __name__ == '__main__':
    model = RNN()
    print(model.state_dict())