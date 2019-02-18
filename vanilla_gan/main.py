import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

n_samples = 1000000
n_features = 200
n_noise_features = 100
epochs = 10
k = 1
batch_size = 64

train = np.random.randn(n_samples, n_features)
print(train.shape)


class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        if len(layer_sizes) > 1:
            for i, layer in enumerate(layer_sizes[:-1]):
                self.layers.append(nn.Linear(layer, layer_sizes[i+1]))
        self.output = nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return F.relu(self.output(x))

discriminator = MLP(n_features, [128, 64, 32], 1)
generator = MLP(n_noise_features, [128, 256, 256], n_features)

print('Discriminator\n{}\n\nGenerator\n{}'.format(discriminator, generator))
print(train.shape)
for e in range(epochs):
    for i in range(k):
        noises = np.random.rand(batch_size, n_noise_features)
        idx = np.random.randint(n_samples, size=batch_size)
        print(idx)
        print(train.shape)
        batch = train[[1,2], :]
        print(batch.size)