from torch import nn
from spikingjelly.activation_based import encoding, functional


class BaseJellyNet(nn.Module):
    def __init__(self,
                 n_input: int = 1,
                 n_output: int = 10,
                 n_steps: int = 5,
                 encoding_method: str = 'direct'):
        """Constructor of BaseJellyNet Class

        Args:
            n_input (int, optional): The number of input dimensions. Defaults to 1.
            n_output (int, optional): The number of output neurons. Defaults to 10.
            n_steps (int, optional): The number of timesteps to encode the input tensor. Defaults to 5.
            encoding_method (str, optional): The neural coding method to perform. Defaults to 'direct'.
        """
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_steps = n_steps
        self.encoding_method = encoding_method
        self.encoder = self.get_encoder()

    def get_encoder(self):
        available_methods = ['direct', 'poisson', 'latency', 'phase']
        if self.encoding_method not in available_methods:
            raise ValueError(
                'encoding_method must be of {}'.format(available_methods))
        if self.encoding_method == 'direct':
            return None
        elif self.encoding_method == 'poisson':
            return encoding.PoissonEncoder()
        elif self.encoding_method == 'latency':
            return encoding.LatencyEncoder(T=self.n_steps)

    def encode_input(self, x):
        if self.encoder:
            x = self.encoder(x)
            if issubclass(type(self.encoder), encoding.StatefulEncoder):
                functional.reset_net(self.encoder)
        return x
