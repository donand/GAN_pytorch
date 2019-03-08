import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import load, Loader
import os
import sys
import datetime
import shutil
import matplotlib.pyplot as plt
import pandas as pd

image_size = (3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, fully_connected_size,
                 dropout_prob=0.3):
        super(Discriminator, self).__init__()
        self.flattened_size = 64 * (image_size[1]//2//2//2) * (image_size[2]//2//2//2)
        self.conv_block = nn.Sequential(
            # input is (3, 32, 32)
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (16, 16, 16)
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (32, 8, 8)
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (64, 4, 4)
            nn.Conv2d(64, 1, 3, padding=0, stride=1),
            nn.Sigmoid()
        )
        '''self.conv1 = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(16, 0.8)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(32, 0.8)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flattened_size = 64 * (image_size[1]//2//2//2) * (image_size[2]//2//2//2)
        self.fully1 = nn.Linear(self.flattened_size, fully_connected_size)
        self.output = nn.Linear(fully_connected_size, output_size)
        self.sigmoid = nn.Sigmoid()'''

    def forward(self, x):
        '''x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(self.bn2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(self.bn3(x)), negative_slope=0.2)
        x = x.view(-1, self.flattened_size)
        x = self.dropout(F.relu(self.fully1(x)))
        return self.sigmoid(self.output(x))'''
        x = self.conv_block(x)
        return x.view(-1, 1)


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor)
        return x


class Generator(nn.Module):
    def __init__(self, input_size, output_channels, fully_connected_size,
                 dropout_prob=0.3):
        super(Generator, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(input_size, 128, 3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, output_channels, 3, stride=2, padding=1),
            nn.Tanh(),
        )

        '''self.init_size = (image_size[1] // 4, image_size[2] // 4)
        self.linear1 = nn.Linear(input_size, 128 * self.init_size[0] * self.init_size[1])
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            Interpolate(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            Interpolate(2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Tanh()
        )'''

    def forward(self, x):
        '''x = self.linear1(x)
        x = x.view(x.shape[0], 128, *self.init_size)'''
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.conv_block(x)
        return x


def discriminator_loss(output_discriminator, output_generator):
    return - torch.mean(torch.log(output_discriminator.squeeze()) + torch.log(1 - output_generator.squeeze()))


def generator_loss(output_generator):
    return - torch.mean(torch.log(output_generator.squeeze()))


def generate_data(n_samples):
    #return torch.from_numpy(np.random.exponential(size=(n_samples, n_features))).type(dtype=torch.FloatTensor)
    return torch.from_numpy(np.random.randn(n_samples, n_features) + 3).type(dtype=torch.FloatTensor)


def get_train_loader(batch_size):
    data_path = 'data/img_align_celeba/'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return iter(train_loader), train_loader


def load_cifar(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = torchvision.datasets.CIFAR10(
        'data', train=True,
        download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return iter(train_loader), train_loader


def get_next_batch(iterator, train_loader):
    batch = next(iterator, None)
    if batch is None:
        iterator = iter(train_loader)
        batch = next(iterator, None)
    return batch[0].to(device), batch[1].to(device), iterator, train_loader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

'''images, labels = iter(get_train_loader(batch_size)).next()
imshow(images[0].numpy())
plt.show()'''

# Create the result directory
result_dir = '{}/'.format(datetime.datetime.now().strftime('%y-%m-%d_%H-%M'))
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
else:
    print('The result directory {} already exists, ABORTING')
    sys.exit(-1)

n_samples = n_samples // batch_size * batch_size
#train = torch.from_numpy(np.random.randn(n_samples, n_features) + 1).type(dtype=torch.FloatTensor)
#train = generate_data(n_samples)
#print(train.shape)

discriminator = Discriminator(image_size, 1, 1024).to(device)
generator = Generator(n_noise_features, image_size[0], 1024).to(device)

print('Discriminator\n{}\n\nGenerator\n{}'.format(discriminator, generator))
#print(train.shape)

disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
loss = torch.nn.BCELoss()

#iterator, train_loader = get_train_loader(batch_size)
iterator, train_loader = load_cifar(batch_size)

disc_losses, gen_losses, gen_means, gen_stds = [], [], [], []

for e in range(epochs):
    if e % print_every == 0:
        print('Epoch {}'.format(e))
    #########################
    # Train the discriminator
    #########################
    temp, temp2 = [], []
    discriminator.train()
    generator.eval()
    for i in range(k):
        disc_optimizer.zero_grad()
        noises = torch.from_numpy(np.random.rand(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
        '''idx = np.random.randint(n_samples, size=batch_size)
        batch = train[idx, :]'''
        images, labels, iterator, train_loader = get_next_batch(iterator, train_loader)
        # Compute output of both the discriminator and generator
        disc_output = discriminator(images)
        gen_output = discriminator(generator(noises))
        # Compute the discriminator loss
        #disc_loss = discriminator_loss(disc_output, gen_output)
        disc_loss = loss(disc_output, torch.ones(disc_output.shape[0], 1).to(device))
        gen_loss = loss(gen_output, torch.zeros(gen_output.shape[0], 1).to(device))
        disc_loss = (disc_loss + gen_loss) / 2
        disc_losses.append(disc_loss.item())
        temp.append(gen_loss.item())
        temp2.append(np.mean(gen_output.cpu().detach().numpy()))
        # Perform the optimization step for the discriminator
        disc_loss.backward()
        disc_optimizer.step()
    #print(np.mean(temp), np.mean(temp2))

    #######################
    # Train the generator
    #######################
    generator.train()
    discriminator.eval()
    temp3 = []
    for i in range(gen_steps):
        gen_optimizer.zero_grad()
        noises = torch.from_numpy(np.random.rand(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
        gen_images = generator(noises)
        gen_output = discriminator(gen_images)
        #print(torch.mean(gen_output).item())
        # Compute the generator loss
        #gen_loss = generator_loss(gen_output)
        gen_loss = loss(gen_output, torch.ones(gen_output.shape[0], 1).to(device))
        # Perform the optimization step for the generator
        gen_loss.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss.item())
        temp3.append(np.mean(gen_output.cpu().detach().numpy()))
    #print('------------', gen_loss.item(), np.mean(temp3))
    #print([x.grad for x in list(generator.parameters())])
    if e % print_every == 0:
        print('D loss: {:.5f}\tG loss: {:.5f}'.format(np.mean(disc_losses[-k:]), np.mean(gen_losses[-gen_steps:])))


discriminator.eval()
generator.eval()
test, labels, iterator, train_loader = get_next_batch(iterator, train_loader)
noises = torch.from_numpy(np.random.rand(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
disc_output = discriminator(test).detach().to('cpu')
gen_output = generator(noises).detach()
print(disc_output.shape, gen_output.to('cpu').shape)
disc_accuracy = np.mean(disc_output.squeeze().detach().numpy())
gen_accuracy = np.mean(discriminator(gen_output).to('cpu').squeeze().detach().numpy())
print('Discriminator accuracy on real data: {}\nDiscriminator accuracy on generated data: {}'.format(disc_accuracy, 1 - gen_accuracy))


# Plot the real and generated distributions
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 10))
# display 20 images
for idx in np.arange(min(batch_size, 20)):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(gen_output[idx].cpu().numpy())
    ax.set_title('asd')
plt.show()

for i in range(min(batch_size, 5)):
    imshow(gen_output[i].cpu().numpy())
    plt.savefig('{}{}'.format(result_dir, i))

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
disc_dict['n_features'] = n_features
torch.save(disc_dict, '{}discriminator.pt'.format(result_dir))

gen_dict = generator.state_dict()
gen_dict['n_features'] = n_features
torch.save(gen_dict, '{}generator.pt'.format(result_dir))

# Copy the config.yml to result directory
shutil.copy2('config.yml', '{}config.yml'.format(result_dir))