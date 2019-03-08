import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import load, Loader
import os
import sys
import datetime
import shutil



class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        if len(layer_sizes) > 1:
            for i, layer in enumerate(layer_sizes[:-1]):
                self.layers.append(nn.Linear(layer, layer_sizes[i+1]))
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(layer_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.tanh(layer(x)))
        return self.sigmoid(self.output(x))


class Generator(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, dropout_prob=0.5):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        if len(layer_sizes) > 1:
            for i, layer in enumerate(layer_sizes[:-1]):
                self.layers.append(nn.Linear(layer, layer_sizes[i+1]))
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(layer_sizes[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(F.relu(layer(x)))
        return self.output(x)

def discriminator_loss(output_discriminator, output_generator):
    return - torch.mean(torch.log(output_discriminator.squeeze()) + torch.log(1 - output_generator.squeeze()))

def generator_loss(output_generator):
    return - torch.mean(torch.log(output_generator.squeeze()))

def generate_data(n_samples):
    #return torch.from_numpy(np.random.exponential(size=(n_samples, n_features))).type(dtype=torch.FloatTensor)
    return torch.from_numpy(np.sort(np.random.randn(n_samples, n_features)) + 3).type(dtype=torch.FloatTensor)

def generate_noise(n_samples):
    return torch.from_numpy(np.sort(np.random.rand(batch_size, n_noise_features))).type(dtype=torch.FloatTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

# Load hyperparameters
stream = open('config.yml', 'r')
config = load(stream, Loader)

n_samples = config['n_samples']
n_features = config['n_features']
n_noise_features = config['n_noise_features']
epochs = config['epochs']
k = config['k']
gen_steps = config['gen_steps']
batch_size = config['batch_size']
print_every = config['print_every']
discriminator_layers = config['discriminator_layers']
generator_layers = config['generator_layers']

result_dir = '{}/'.format(datetime.datetime.now().strftime('%y-%m-%d_%H-%M'))
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
else:
    print('The result directory {} already exists, ABORTING')
    sys.exit(-1)

n_samples = n_samples // batch_size * batch_size
#train = torch.from_numpy(np.random.randn(n_samples, n_features) + 1).type(dtype=torch.FloatTensor)
train = generate_data(n_samples)
print(train.shape)

discriminator = MLP(n_features, discriminator_layers, 1).to(device)
generator = Generator(n_noise_features, generator_layers, n_features).to(device)

print('Discriminator\n{}\n\nGenerator\n{}'.format(discriminator, generator))
print(train.shape)

disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
loss = torch.nn.BCELoss()

disc_losses, gen_losses, gen_means, gen_stds = [], [], [], []

#i = 0
for e in range(epochs):
    if e % print_every == 0:
        print('Epoch {}'.format(e))
    discriminator.train()
    generator.eval()
    #########################
    # Train the discriminator
    #########################
    for i in range(k):
        disc_optimizer.zero_grad()
        noises = generate_noise(batch_size).to(device)
        '''idx = np.random.randint(n_samples, size=batch_size)
        batch = train[idx, :]'''
        batch = generate_data(batch_size).to(device)
        # Compute output of both the discriminator and generator
        disc_output = discriminator(batch)
        gen_output = discriminator(generator(noises))
        # Compute the discriminator loss
        #disc_loss = discriminator_loss(disc_output, gen_output)
        disc_loss = loss(disc_output, torch.ones(batch_size, 1).to(device))
        gen_loss = loss(gen_output, torch.zeros(batch_size, 1).to(device))
        disc_loss = (disc_loss + gen_loss) / 2
        disc_losses.append(disc_loss.item())
        # Perform the optimization step for the discriminator
        disc_loss.backward()
        disc_optimizer.step()

    #######################
    ### Train the generator
    #######################
    generator.train()
    discriminator.eval()
    for i in range(gen_steps):
        gen_optimizer.zero_grad()
        noises = generate_noise(batch_size).to(device)
        generated = generator(noises)
        gen_output = discriminator(generated)
        #print(torch.mean(gen_output).item())
        # Compute the generator loss
        #gen_loss = generator_loss(gen_output)
        gen_loss = loss(gen_output, torch.ones(batch_size, 1).to(device))
        # Perform the optimization step for the generator
        gen_loss.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss.item())
        gen_means.append(torch.mean(generated.squeeze().cpu()))
        gen_stds.append(torch.std(generated.squeeze().cpu()))
    #print([x.grad for x in list(generator.parameters())])
    if e % print_every == 0:
        print('D loss: {:.5f}\tG loss: {:.5f}'.format(np.mean(disc_losses[-k:]), np.mean(gen_losses[-gen_steps:])))


discriminator.eval()
generator.eval()
test = generate_data(n_samples).to(device)
noises = generate_noise(2000).detach().to(device)
disc_output = discriminator(test[:2000]).detach().to('cpu')
gen_output = generator(noises).detach()
print(disc_output.shape, gen_output.to('cpu').shape)
disc_accuracy = np.mean(disc_output.squeeze().detach().numpy())
gen_accuracy = np.mean(discriminator(gen_output).to('cpu').squeeze().detach().numpy())
print('Discriminator accuracy on real data: {}\nDiscriminator accuracy on generated data: {}'.format(disc_accuracy, 1 - gen_accuracy))


# Plot the real and generated distributions
test = test.cpu()
gen_output = gen_output.cpu()
if n_features >= 25:
    fig, ax = plt.subplots(5, 5, figsize=(15,15))
    plt.title('Generated vs Real Distributions')
    for i in range(5):
        for j in range(5):
            idx = i*5 + j
            sns.distplot(test[:2000, idx], label='Real - dim 0', ax=ax[i][j])
            sns.distplot(gen_output[:, idx], label='Generated - dim 0', ax=ax[i][j])
            ax[i][j].xlabel('Samples')
    plt.legend()
    plt.savefig('{}generated_vs_real_distribution'.format(result_dir), dpi=200)
else:
    plt.title('Generated vs Real Distributions')
    sns.distplot(test[:2000, 0], label='Real - dim 0')
    sns.distplot(gen_output[:, 0], label='Generated - dim 0')
    plt.xlabel('Samples')
    plt.legend()
    plt.savefig('{}generated_vs_real_distribution'.format(result_dir), dpi=200)

fig = plt.figure()
sns.distplot(gen_output[:, 0])
plt.savefig('{}generated'.format(result_dir), dpi=200)

# PLot the generator and discriminator losses
fig = plt.figure()
plt.title('Discriminator Loss')
rolling = pd.Series(disc_losses).rolling(print_every).mean()
plt.plot(range(len(rolling)), rolling)
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig('{}discriminator_loss'.format(result_dir), dpi=200)
fig = plt.figure()
plt.title('Generator Loss')
rolling = pd.Series(gen_losses).rolling(print_every).mean()
plt.plot(range(len(rolling)), rolling)
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig('{}generator_loss'.format(result_dir), dpi=200)

fig = plt.figure()
plt.title('Mean')
rolling = pd.Series(gen_means).rolling(print_every).mean()
plt.plot(range(len(rolling)), rolling)
plt.xlabel('Training steps')
plt.ylabel('Mean')
plt.savefig('{}mean'.format(result_dir), dpi=200)
fig = plt.figure()
plt.title('Standard Deviation')
rolling = pd.Series(gen_stds).rolling(print_every).mean()
plt.plot(range(len(rolling)), rolling)
plt.xlabel('Training steps')
plt.ylabel('Std')
plt.savefig('{}std'.format(result_dir), dpi=200)

# Save the models
disc_dict = discriminator.state_dict()
disc_dict['layers'] = discriminator_layers
disc_dict['n_features'] = n_features
torch.save(disc_dict, '{}discriminator.pt'.format(result_dir))

gen_dict = generator.state_dict()
gen_dict['layers'] = generator_layers
gen_dict['n_features'] = n_features
torch.save(gen_dict, '{}generator.pt'.format(result_dir))

# Copy the config.yml to result directory
shutil.copy2('config.yml', '{}config.yml'.format(result_dir))