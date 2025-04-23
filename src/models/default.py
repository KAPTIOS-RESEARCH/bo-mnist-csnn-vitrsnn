from torch import nn
from spikingjelly.activation_based import surrogate, neuron, layer, functional
from .base import BaseJellyNet


class DefaultSNN(BaseJellyNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super().__init__(in_channels, out_channels, n_steps=5, encoding_method='direct')

        self.net = nn.Sequential(
            layer.Conv2d(in_channels, 8, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(8),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(8, 16, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(16),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(16 * 7 * 7, 16 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(16 * 4 * 4, out_channels, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x
