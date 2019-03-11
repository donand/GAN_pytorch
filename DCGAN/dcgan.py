import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import load, Loader
import os
import sys
import datetime
import shutil
import pandas as pd
import time

image_size = (1, 32, 32)
grayscale = True


class Discriminator(nn.Module):
    def __init__(self, input_channels, nf):
        super(Discriminator, self).__init__()
        self.flattened_size = 64 * (image_size[1]//2//2//2) * (image_size[2]//2//2//2)
        self.conv_block = nn.Sequential(
            # input is (3, 32, 32)
            nn.Conv2d(input_channels, nf, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf, 16, 16)
            nn.Conv2d(nf, nf * 2, 4, padding=1, stride=2),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*2, 8, 8)
            nn.Conv2d(nf * 2, nf * 4, 4, padding=1, stride=2),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*4, 4, 4)
            nn.Conv2d(nf * 4, 1, 4, padding=0, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x.view(-1, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Generator(nn.Module):
    def __init__(self, input_size, output_channels, nf=128):
        super(Generator, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(input_size, nf*4, 4, stride=1, padding=0),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf, output_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.conv_block(x)
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def generator_loss(output_generator):
    return - torch.mean(torch.log(output_generator.squeeze()))


def plot_results(result_dir):
    fig = plt.figure()
    plt.title('Discriminator Loss')
    rolling = pd.Series(disc_losses).rolling(rolling_window).mean()
    plt.plot(range(len(rolling)), rolling)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.savefig('{}discriminator_loss'.format(result_dir), dpi=200)
    plt.close(fig)
    fig = plt.figure()
    plt.title('Generator Loss')
    rolling = pd.Series(gen_losses).rolling(rolling_window).mean()
    plt.plot(range(len(rolling)), rolling)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.savefig('{}generator_loss'.format(result_dir), dpi=200)
    plt.close(fig)


def checkpoint(disc, gen, epoch):
    check_dir = '{}checkpoint_ep{}/'.format(result_dir, epoch)
    if not os.path.isdir(check_dir):
        os.makedirs(check_dir)
    disc_dict = discriminator.state_dict()
    torch.save(disc_dict, '{}discriminator.pt'.format(check_dir))
    gen_dict = generator.state_dict()
    torch.save(gen_dict, '{}generator.pt'.format(check_dir))
    plot_results(check_dir)

    noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
    gen_output = generator(noises).detach()
    fig = plt.figure()
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        imshow(gen_output[idx].cpu().numpy())
    plt.savefig('{}generated'.format(check_dir), dpi=200)
    plt.close(fig)


def generate_frame(disc, gen, epoch):
    noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
    gen_output = generator(noises).detach()
    fig = plt.figure()
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        imshow(gen_output[idx].cpu().numpy())
    fig.suptitle('Epoch {}'.format(epoch + 1))
    plt.savefig('{}frame_{}'.format(video_dir, epoch), dpi=200)
    plt.close(fig)


'''def get_train_loader(batch_size):
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
    return iter(train_loader), train_loader'''


def load_dataset(batch_size, dataset, image_size):
    if dataset not in ['MNIST', 'CIFAR10', 'CELEBA']:
        print('Dataset not known: {}'.format(dataset))
        sys.exit(-1)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == 'MNIST':
        train_data = torchvision.datasets.MNIST(
            'data', train=True,
            download=True, transform=transform
        )
        test_data = torchvision.datasets.MNIST(
            'data', train=False,
            download=True, transform=transform
        )
    elif dataset == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(
            'data', train=True,
            download=True, transform=transform
        )
        test_data = torchvision.datasets.CIFAR10(
            'data', train=False,
            download=True, transform=transform
        )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return iter(train_loader), train_loader, test_loader


'''def get_next_batch(iterator, train_loader):
    batch = next(iterator, None)
    if batch is None:
        iterator = iter(train_loader)
        batch = next(iterator, None)
    return batch[0].to(device), batch[1].to(device), iterator, train_loader'''


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    if grayscale:
        plt.imshow(np.squeeze(img), cmap='gray')
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load hyperparameters
stream = open('config.yml', 'r')
config = load(stream, Loader)

n_noise_features = config['n_noise_features']
epochs = config['epochs']
k = config['k']
gen_steps = config['gen_steps']
batch_size = config['batch_size']
print_every = config['print_every']
checkpoints = config['checkpoints']
rolling_window = config['rolling_window']
discriminator_filters = config['discriminator_filters']
generator_filters = config['generator_filters']
discriminator_label_noise = config['discriminator_label_noise']
discriminator_input_noise = config['discriminator_input_noise']

# Create the result directory
result_dir = '{}_e{}_d{}_g{}/'.format(
    datetime.datetime.now().strftime('%y-%m-%d_%H-%M'),
    epochs,
    discriminator_filters,
    generator_filters
)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
else:
    print('The result directory {} already exists, ABORTING')
    sys.exit(-1)

# Copy the config.yml to result directory
shutil.copy2('config.yml', '{}config.yml'.format(result_dir))

# Create the directory for the frames of the epochs
video_dir = '{}video/'.format(result_dir)
if not os.path.isdir(video_dir):
    os.makedirs(video_dir)

discriminator = Discriminator(image_size[0], discriminator_filters).to(device)
generator = Generator(n_noise_features, image_size[0], generator_filters).to(device)
discriminator.weight_init(mean=0.0, std=0.02)
generator.weight_init(mean=0.0, std=0.02)

print('Discriminator\n{}\n\nGenerator\n{}'.format(discriminator, generator))

disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss = torch.nn.BCELoss()

# iterator, train_loader = get_train_loader(batch_size)
iterator, train_loader, test_loader = load_dataset(batch_size,
                                                   'MNIST',
                                                   image_size[1])
images = iterator.next()[0].numpy()
print('Image size: {}'.format(images[0].shape))
# Plot images
fig = plt.figure()
for idx in np.arange(16):
    ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
plt.show()
plt.close(fig)

# Plot images with noise
input_noise = torch.randn(*images[0].shape) * 0.1
fig = plt.figure()
for idx in np.arange(10):
    ax = fig.add_subplot(5, 2, idx+1, xticks=[], yticks=[])
    if idx % 2 == 0:
        imshow(images[idx])
    else:
        imshow(images[idx-1] + input_noise)
plt.show()
plt.close(fig)

disc_losses, gen_losses = [], []

for e in range(epochs):
    if e % print_every == 0:
        print('Epoch {}'.format(e))
    start = time.time()
    epoch_dlosses, epoch_glosses = [], []
    for images, _ in train_loader:
        images = images.to(device)
        noise_factor = (epochs - e) / epochs
        #########################
        # Train the discriminator
        #########################
        for i in range(k):
            disc_optimizer.zero_grad()
            noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
            # Apply noise to input images
            if discriminator_input_noise:
                input_noise = torch.randn(*images.shape) * 0.1 * noise_factor
                images = images + input_noise
            # Compute output of both the discriminator and generator
            disc_output = discriminator(images)
            gen_output = discriminator(generator(noises))
            # Apply noise to labels
            disc_label_noise = torch.ones(images.shape[0], 1).to(device)
            gen_label_noise = torch.zeros(batch_size, 1).to(device)
            if discriminator_label_noise:
                disc_label_noise -= (torch.rand(images.shape[0], 1) * 0.2 * noise_factor).to(device)
                gen_label_noise += (torch.rand(batch_size, 1) * 0.2 * noise_factor).to(device)
            # Compute the discriminator loss
            disc_loss = loss(disc_output, disc_label_noise)
            gen_loss = loss(gen_output, gen_label_noise)
            disc_loss = disc_loss + gen_loss
            # Perform the optimization step for the discriminator
            disc_loss.backward()
            disc_optimizer.step()
            # Save the loss
            disc_losses.append(disc_loss.item())
            epoch_dlosses.append(disc_loss.item())

        #######################
        # Train the generator
        #######################
        for i in range(gen_steps):
            gen_optimizer.zero_grad()
            noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
            gen_images = generator(noises)
            gen_output = discriminator(gen_images)
            # Compute the generator loss
            gen_loss = loss(gen_output, torch.ones(batch_size, 1).to(device))
            # Perform the optimization step for the generator
            gen_loss.backward()
            gen_optimizer.step()
            # Save the loss
            gen_losses.append(gen_loss.item())
            epoch_glosses.append(gen_loss.item())
        #print('------------', gen_loss.item(), np.mean(temp3))
        #print([x.grad for x in list(generator.parameters())])
    generate_frame(discriminator, generator, e)
    if e % print_every == 0:
        print('D loss: {:.5f}\tG loss: {:.5f}\tTime: {:.0f}'.format(np.mean(epoch_dlosses), np.mean(epoch_glosses), time.time() - start))
    if e != 0 and e % checkpoints == 0:
        checkpoint(discriminator, generator, e)


disc_accs, gen_accs = [], []
for test, _ in test_loader:
    test = test.to(device)
    noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(dtype=torch.FloatTensor).to(device)
    disc_output = discriminator(test).detach().to('cpu')
    gen_output = generator(noises).detach()
    #print(disc_output.shape, gen_output.to('cpu').shape)
    disc_accs.append(np.mean(disc_output.squeeze().numpy()))
    gen_accs.append(np.mean(discriminator(gen_output).to('cpu').squeeze().detach().numpy()))

print('Discriminator accuracy on real data: {}\nDiscriminator accuracy on generated data: {}'.format(np.mean(disc_accs), 1 - np.mean(gen_accs)))


# Plot 16 generated images
fig = plt.figure()
for idx in np.arange(16):
    ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
    imshow(gen_output[idx].cpu().numpy())
plt.savefig('{}generated'.format(result_dir), dpi=200)
plt.close(fig)

# Plot 5 generated images in separate files
for i in range(min(batch_size, 5)):
    fig = plt.figure()
    imshow(gen_output[i].cpu().numpy())
    plt.savefig('{}{}'.format(result_dir, i))
    plt.close(fig)

# Plot the generator and discriminator losses
plot_results(result_dir)

# Save the models
disc_dict = discriminator.state_dict()
torch.save(disc_dict, '{}discriminator.pt'.format(result_dir))

gen_dict = generator.state_dict()
torch.save(gen_dict, '{}generator.pt'.format(result_dir))