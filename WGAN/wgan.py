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
from tensorboardX import SummaryWriter

image_size = (3, 64, 64)
grayscale = False
DATA_FOLDER = '../data/'


class Discriminator(nn.Module):
    def __init__(self, input_channels, nf):
        super(Discriminator, self).__init__()
        self.flattened_size = 64 * \
            (image_size[1]//2//2//2) * (image_size[2]//2//2//2)
        self.conv_block = nn.Sequential(
            # input is (3, 32, 32)
            nn.Conv2d(input_channels, nf, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf, 16, 16)
            nn.Conv2d(nf, nf * 2, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*2, 8, 8)
            nn.Conv2d(nf * 2, nf * 4, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(nf * 4, nf * 8, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*4, 4, 4)
            nn.Conv2d(nf * 8, 1, 4, padding=0, stride=1, bias=False),
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
            nn.ConvTranspose2d(input_size, nf*8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*8, nf*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf, output_channels, 4, stride=2, padding=1, bias=False),
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
    plt.savefig('{}discriminator_loss'.format(result_dir), dpi=300)
    plt.close(fig)
    fig = plt.figure()
    plt.title('Generator Loss')
    rolling = pd.Series(gen_losses).rolling(rolling_window).mean()
    plt.plot(range(len(rolling)), rolling)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.savefig('{}generator_loss'.format(result_dir), dpi=300)
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

    noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
        dtype=torch.FloatTensor).to(device)
    gen_output = generator(noises).detach()
    fig = plt.figure(figsize=(10,10))
    imshow(gen_output.cpu())
    plt.title('Epoch {}'.format(epoch+1))
    plt.savefig('{}generated'.format(check_dir), dpi=300)
    plt.close(fig)


def generate_frame(disc, gen, epoch):
    noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
        dtype=torch.FloatTensor).to(device)
    gen_output = generator(noises).detach()
    fig = plt.figure(figsize=(10, 10))
    imshow(gen_output.cpu())
    fig.suptitle('Epoch {}'.format(epoch + 1))
    plt.savefig('{}frame_{}'.format(video_dir, epoch), dpi=300)
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
    if dataset not in ['MNIST', 'CIFAR10', 'CELEBA', 'POKEMON']:
        print('Dataset not known: {}'.format(dataset))
        sys.exit(-1)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == 'MNIST':
        train_data = torchvision.datasets.MNIST(
            DATA_FOLDER, train=True,
            download=True, transform=transform
        )
        test_data = torchvision.datasets.MNIST(
            DATA_FOLDER, train=False,
            download=True, transform=transform
        )
    elif dataset == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(
            DATA_FOLDER, train=True,
            download=True, transform=transform
        )
        test_data = torchvision.datasets.CIFAR10(
            DATA_FOLDER, train=False,
            download=True, transform=transform
        )
    elif dataset == 'CELEBA':
        data_path = '{}img_align_celeba/'.format(DATA_FOLDER)
        train_data = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )
    elif dataset == 'POKEMON':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_path = '{}pokemon/'.format(DATA_FOLDER)
        train_data = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    if dataset != 'CELEBA' and dataset != 'POKEMON':
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
    else:
        test_loader = train_loader
    return train_loader, test_loader


'''def get_next_batch(iterator, train_loader):
    batch = next(iterator, None)
    if batch is None:
        iterator = iter(train_loader)
        batch = next(iterator, None)
    return batch[0].to(device), batch[1].to(device), iterator, train_loader'''


def imshow(images):
    images = images / 2 + 0.5  # unnormalize
    grid = torchvision.utils.make_grid(images)
    if grayscale:
        plt.imshow(grid.squeeze(), cmap='gray')
    else:
        plt.imshow(grid.permute(1, 2, 0))
    
    


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load hyperparameters
stream = open('config.yml', 'r')
config = load(stream, Loader)

dataset = config['dataset']
n_noise_features = config['n_noise_features']
epochs = config['epochs']
disc_steps = config['disc_steps']
gen_steps = config['gen_steps']
batch_size = config['batch_size']
print_every = config['print_every']
checkpoints = config['checkpoints']
rolling_window = config['rolling_window']
discriminator_filters = config['discriminator_filters']
generator_filters = config['generator_filters']
discriminator_label_noise = config['discriminator_label_noise']
discriminator_input_noise = config['discriminator_input_noise']
resume_training = config['resume_training']

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

writer = SummaryWriter(log_dir='{}tensorboard'.format(result_dir))

discriminator = Discriminator(image_size[0], discriminator_filters).to(device)
generator = Generator(
    n_noise_features, image_size[0], generator_filters).to(device)
discriminator.weight_init(mean=0.0, std=0.02)
generator.weight_init(mean=0.0, std=0.02)

print('Discriminator\n{}\n\nGenerator\n{}'.format(discriminator, generator))

disc_optimizer = torch.optim.RMSprop(
    discriminator.parameters(), lr=0.00005)
gen_optimizer = torch.optim.RMSprop(
    generator.parameters(), lr=0.00005)

# iterator, train_loader = get_train_loader(batch_size)
train_loader, test_loader = load_dataset(batch_size,
                                         dataset,
                                         image_size[1])

images = iter(train_loader).next()[0]
img = images.numpy()
print('Max: {}\tMin: {}\tMean: {}\tStd: {}'.format(
    np.max(img),
    np.min(img),
    (np.mean(img[:,0]), np.mean(img[:,1]), np.mean(img[:,2])),
    (np.std(img[:,0]), np.std(img[:,1]), np.std(img[:,2]))
))
print('Image size: {}'.format(images[0].shape))
fig = plt.figure(figsize=(10,10))
imshow(images)
plt.show()
plt.close(fig)

# Plot images with noise
'''images = images.numpy()
input_noise = np.random.randn(*images[0].shape) * 0.07
fig = plt.figure()
for idx in np.arange(10):
    ax = fig.add_subplot(5, 2, idx+1, xticks=[], yticks=[])
    if idx % 2 == 0:
        imshow(images[idx])
    else:
        imshow(images[idx-1] + input_noise)
plt.show()
plt.close(fig)'''

disc_losses, gen_losses = [], []
gen_iterations = 0
steps = 0

for e in range(epochs):
    if e % print_every == 0:
        print('Epoch {}'.format(e))
    start = time.time()
    epoch_dlosses, epoch_glosses = [], []
    train_iterator = iter(train_loader)
    i = 0
    while i < len(train_loader):
        noise_factor = (epochs - e) / epochs
        #########################
        # Train the discriminator
        #########################
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True
        # train the discriminator disc_steps times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            disc_steps = 100
        else:
            disc_steps = config['disc_steps']
        j = 0
        while j < disc_steps and i < len(train_loader):
            j += 1
            i += 1
            images, _ = train_iterator.next()
            images = images.to(device)
            common_batch_size = min(batch_size, images.shape[0])
            disc_optimizer.zero_grad()
            noises = torch.from_numpy(np.random.randn(common_batch_size, n_noise_features)).type(
                dtype=torch.FloatTensor).to(device)
            '''# Apply noise to input images
            if discriminator_input_noise:
                input_noise_d = torch.randn(
                    *images.shape).to(device) * 0.07 * noise_factor
                input_noise_g = torch.randn(common_batch_size, 1).to(
                    device) * 0.07 * noise_factor
                images = images + input_noise_d
                noises = noises + input_noise_g'''
            # Compute output of both the discriminator and generator
            disc_output = discriminator(images)
            gen_output = discriminator(generator(noises))
            disc_output.backward(torch.ones(common_batch_size, 1).to(device))
            gen_output.backward(- torch.ones(common_batch_size, 1).to(device))
            #loss = - torch.mean(disc_output) + torch.mean(gen_output)
            #loss.backward()
            errD = disc_output - gen_output
            disc_optimizer.step()
            # clamp parameters to a cube
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            '''# Apply noise to labels
            disc_label_noise = torch.ones(images.shape[0], 1).to(device)
            gen_label_noise = torch.zeros(batch_size, 1).to(device)
            if discriminator_label_noise:
                disc_label_noise -= (torch.rand(
                    images.shape[0], 1) * 0.2 * noise_factor).to(device)
                gen_label_noise += (torch.rand(batch_size, 1)
                                    * 0.2 * noise_factor).to(device)'''

            # Save the loss
            disc_losses.append(torch.mean(errD).item())
            epoch_dlosses.append(torch.mean(errD).item())
            writer.add_scalar('data/D_loss', torch.mean(errD).item(), steps)
            #disc_losses.append(loss.item())
            #epoch_dlosses.append(loss.item())
            #writer.add_scalar('data/D_loss', loss.item(), steps)
            steps += 1

        #######################
        # Train the generator
        #######################
        #print('Training generator {} {}'.format(gen_iterations, i))
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = False
        gen_optimizer.zero_grad()
        noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
            dtype=torch.FloatTensor).to(device)
        gen_images = generator(noises)
        gen_output = discriminator(gen_images)
        gen_output.backward(torch.ones(batch_size, 1).to(device))
        #loss = - torch.mean(gen_output)
        #loss.backward()
        gen_optimizer.step()
        # Save the loss
        gen_losses.append(torch.mean(gen_output).item())
        epoch_glosses.append(torch.mean(gen_output).item())
        #print('------------', gen_loss.item(), np.mean(temp3))
        #print([x.grad for x in list(generator.parameters())])
        writer.add_scalar('data/G_loss', torch.mean(gen_output).item(), steps)
        gen_iterations += 1
        steps += 1
    if e % print_every == 0:
        generate_frame(discriminator, generator, e)
        print('D loss: {:.5f}\tG loss: {:.5f}\tTime: {:.0f}'.format(
            np.mean(epoch_dlosses), np.mean(epoch_glosses), time.time() - start))
    if e % checkpoints == 0:
        checkpoint(discriminator, generator, e)


print('\nTesting...')
for p in discriminator.parameters():  # reset requires_grad
    p.requires_grad = False
for p in generator.parameters():  # reset requires_grad
    p.requires_grad = False
disc_accs, gen_accs = [], []
for test, _ in train_loader:
    test = test.to(device)
    noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
        dtype=torch.FloatTensor).to(device)
    disc_output = discriminator(test).detach().to('cpu')
    gen_output = generator(noises).detach()
    #print(disc_output.shape, gen_output.to('cpu').shape)
    disc_accs.append(np.mean(disc_output.squeeze().numpy()))
    gen_accs.append(np.mean(discriminator(gen_output).to(
        'cpu').squeeze().detach().numpy()))

print('Discriminator accuracy on real data: {}\nDiscriminator accuracy on generated data: {}'.format(
    np.mean(disc_accs), 1 - np.mean(gen_accs)))


# Plot 16 generated images
fig = plt.figure()
imshow(gen_output.cpu())
plt.title('Results')
plt.savefig('{}generated'.format(result_dir), dpi=300)
plt.close(fig)

# Plot 5 generated images in separate files
'''for i in range(min(batch_size, 5)):
    fig = plt.figure()
    imshow(gen_output[i].cpu().numpy())
    plt.savefig('{}{}'.format(result_dir, i))
    plt.close(fig)'''

# Plot the generator and discriminator losses
plot_results(result_dir)

# Save the models
disc_dict = discriminator.state_dict()
torch.save(disc_dict, '{}discriminator.pt'.format(result_dir))

gen_dict = generator.state_dict()
torch.save(gen_dict, '{}generator.pt'.format(result_dir))
