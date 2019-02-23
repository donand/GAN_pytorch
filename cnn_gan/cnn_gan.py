import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import load, Loader
import os
import sys
import datetime
import shutil


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(0,0)
        self.output = nn.Linear(layer_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.Linear(x)))
        return self.sigmoid(self.output(x))


class Generator(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        if len(layer_sizes) > 1:
            for i, layer in enumerate(layer_sizes[:-1]):
                self.layers.append(nn.Linear(layer, layer_sizes[i+1]))
        self.output = nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)

def discriminator_loss(output_discriminator, output_generator):
    return - torch.mean(torch.log(output_discriminator.squeeze()) + torch.log(1 - output_generator.squeeze()))

def generator_loss(output_generator):
    return - torch.mean(torch.log(output_generator.squeeze()))

def generate_data(n_samples):
    #return torch.from_numpy(np.random.exponential(size=(n_samples, n_features))).type(dtype=torch.FloatTensor)
    return torch.from_numpy(np.random.randn(n_samples, n_features) + 3).type(dtype=torch.FloatTensor)

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

discriminator = MLP(n_features, discriminator_layers, 1)
generator = Generator(n_noise_features, generator_layers, n_features)

print('Discriminator\n{}\n\nGenerator\n{}'.format(discriminator, generator))
print(train.shape)

disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
loss = torch.nn.BCELoss()

disc_losses = []
gen_losses = []

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
        noises = torch.from_numpy(np.random.rand(batch_size, n_noise_features)).type(dtype=torch.FloatTensor)
        '''idx = np.random.randint(n_samples, size=batch_size)
        batch = train[idx, :]'''
        batch = generate_data(batch_size)
        # Compute output of both the discriminator and generator
        disc_output = discriminator(batch)
        gen_output = discriminator(generator(noises))
        # Compute the discriminator loss
        #disc_loss = discriminator_loss(disc_output, gen_output)
        disc_loss = loss(disc_output, torch.ones(batch_size, 1))
        gen_loss = loss(gen_output, torch.zeros(batch_size, 1))
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
        noises = torch.from_numpy(np.random.rand(batch_size, n_noise_features)).type(dtype=torch.FloatTensor)
        gen_output = discriminator(generator(noises))
        #print(torch.mean(gen_output).item())
        # Compute the generator loss
        #gen_loss = generator_loss(gen_output)
        gen_loss = loss(gen_output, torch.ones(batch_size, 1))
        # Perform the optimization step for the generator
        gen_loss.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss.item())
    #print([x.grad for x in list(generator.parameters())])
    if e % print_every == 0:
        print('D loss: {:.5f}\tG loss: {:.5f}'.format(np.mean(disc_losses[-k:]), np.mean(gen_losses[-gen_steps:])))


discriminator.eval()
generator.eval()
test = generate_data(n_samples)
noises = torch.from_numpy(np.random.rand(2000, n_noise_features)).type(dtype=torch.FloatTensor).detach()
disc_output = discriminator(test[:2000]).detach()
gen_output = generator(noises).detach()
print(disc_output.shape, gen_output.shape)
disc_accuracy = np.mean(disc_output.squeeze().detach().numpy())
gen_accuracy = np.mean(discriminator(gen_output).squeeze().detach().numpy())
print('Discriminator accuracy on real data: {}\nDiscriminator accuracy on generated data: {}'.format(disc_accuracy, 1 - gen_accuracy))


# Plot the real and generated distributions
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

# PLot the generator and discriminator losses
fig = plt.figure()
plt.title('Discriminator Loss')
plt.plot(range(len(disc_losses)), disc_losses)
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig('{}discriminator_loss'.format(result_dir), dpi=200)
fig = plt.figure()
plt.title('Generator Loss')
plt.plot(range(len(gen_losses)), gen_losses)
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig('{}generator_loss'.format(result_dir), dpi=200)

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