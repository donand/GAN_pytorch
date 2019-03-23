# GAN in PyTorch
In this repository I implement several versions of Generative Adversarial Networks in PyTorch.

All comments and discussions are welcome.

# GAN
In this section I implemented the original version of GAN as described in the paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio.

The implementation can be found in [GAN/gan.py](GAN/gan.py).

## Experiment setup
The target distribution was a Normal distribution with mean=3 and std=1, and the input noise to the generator was sampled from a uniform distribution. Both the target and noise samples are monodimensional, but this can be changed in the config.yml file in order to extend to multiple dimensions.

The discriminator is composed by 3 hidden layers with 16, 16 and 8 neurons respectively, with ReLU activation functions and dropout after each layer with a probability of 0.5. The output layer is composed by only 1 neuron with sigmoid activation function, providing the probability of the input sample belonging to the real distribution and not being generated by the generator.

The generator is composed by 3 hidden layers of sizes 16, 32, 16 relatively, with ReLU activation functions. The output layer has the same size of the number of dimensions of the target samples, so in our case is 1. The output activation function is linear, because we don't want to limit the output values.

## Results
Here some results are reported after training for 5k steps. One step consists in one training step for the discriminator and one for the generator.

As we can see, the generator correctly replicated the original distribution, that was a Gaussan distribution with mean = 3 and std = 1.

A possible next step could be to reduce the size of the networks (reducing the number of neurons per layer or the number of layers) in order to obtain more stable models.

Below are reported some charts of the results of the experiment. As you can see, the generator matched the  real data distribution.<br>
We can also see that the mean and standard deviation of the generated distribution successfully converged to the real ones. We can note that there is still a decreasing trend in the standard deviation, so more training steps could be beneficial.<br><br>

<p align="center"><img src="GAN/results/generated_vs_real_distribution.png" alt="Distributions" width="500" height="400"></p>

Discriminator Loss                                           |  Generator Loss
:-----------------------------------------------------------:|:---------------------------------------:
![Discriminator Loss](GAN/results/discriminator_loss.png) | ![Generator Loss](GAN/results/generator_loss.png)

As we can see, the generator was able to catch the real mean of the data after around 5k steps.<br><br>

Mean of the Generated Distribution                                          |  STD of the Generated Distribution
:-----------------------------------------------------------:|:---------------------------------------:
![Discriminator Loss](GAN/results/mean.png) | ![Generator Loss](GAN/results/std.png)



# Deep Convolutional GAN (DCGAN) - Work in Progress
This will be the implementation of GAN using Deep Convolutional Neural Networks as described in [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Alec Radford, Luke Metz and Soumith Chintala.

I used the same architecture for the discriminator and the generator used in the paper, where the generator is the following

<p align="center"><img src="DCGAN/all_conv_64px_generator.png" alt="Generator Architecture" width="800"></p>
Image taken from [this paper](https://arxiv.org/abs/1511.06434).<br>
Batch Normalization is applied after each convolutional layer. The activation function is Leaky ReLU.

The discriminator has 5 convolutional layers with a kernel size of 4 and stride 2. In this way we don't have to use MaxPooling.<br>
Batch Normalization is applied after each convolutional layer, and the output of the batch normalization layer goes through the Leaky ReLU activation function.<br>
There are no fully connected layers in the network, and at the end a Sigmoid activation is applied.

The implementation can be found in [DCGAN/dcgan.py](DCGAN/dcgan.py).

## CelebA
After training DCGAN for 11 epochs, it achieved pretty decent results.

The discriminator has 128 filters in the first layer, up to 1024 in the last one.<br>
The generator has the same number of filters, starting from 1024 and going down to 128 to the last ConvTranspose layer.

<p align="center">
  <img hspace=20 src="DCGAN/results_celeba/video/celeba.gif" width="300" />
  <img hspace=20 src="DCGAN/results_celeba/video/frame_10.png" width="300" /> 
</p>

# Wasserstein Generative Adversarial Networks (WGAN)
For the implementation of WGAN I followed the paper [Wasserstein GAN](https://arxiv.org/abs/1701.07875) by Martin Arjovsky, Soumith Chintala, Léon Bottou.

The implementation can be found in [WGAN/wgan.py](WGAN/wgan.py).

The main innovation brought by WGAN is the use of the Wasserstein distance as the loss function to evaluate distance between the real distribution of the data and the generated one, instead of the Jensen–Shannon divergence.<br>
Wasserstein distance has a much better behaviour in case of disjoint distributions and when the distributions lay on a low-dimensional manifold. This means that the loss function gives more significative gradients to the critic and the generator.

Another important advantage of using the Wasserstein distance is that the loss function of the critic is now much more informative. In fact, the loss is directly linked to the quality of generated samples, and this means that it's a useful estimator of the performance of the GAN. This was not true for traditional GAN and for DCGAN, where the loss function of the discriminator was dependent on the performance of the generator, and the other way around.

The structure of the critic is similar to the structure of the discriminator in DCGAN, with the exception that Batch Normalization is not used anymore and the output has a linear activation function instead of the Sigmoid.<br>
Batch Normalization cannot be used in the critic network because it introduces dependency between samples in the same batch.

An important contraint for the critic to be an estimator of the Wasserstein distance is to be a 1-Lipschitz continuous function.<br>
This is achieved by clipping the weights of the critic network into a small interval, the one used in the paper is [-0.01, 0.01]. In this way they are enforcing the critic function to be K-Lipschitz continuous.

The generator is the same that was used in DCGAN.

## Results with CelebA Dataset
*Work in Progress*

# Wasserstein GAN with Gradient Penalty (WGAN-GP)
This is the implementation of WGAN-GP that is described in [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville.

The implementation can be found in [WGAN-GP/wgan_gp.py](WGAN-GP/wgan_gp.py).

The main contribution of WGAN-GP is the method used to enforce that the function approximated by the critic is a 1-Lipschitz continuous function. This is achieved by adding a penalty for gradients larger or smaller than 1 in the loss function of the critic.<br>
The weight clipping used in WGAN is a quite hard contraint and it limits the expressive power of the critic network.

The gradient penalty is computed on gradients of the critic w.r.t. a linear combination of real inputs and input noise of the generator. It's a two-way penalty, so it penalizes both gradients larger and smaller than one, to make them equal to one.

The use of Gradient Penalty stabilizes a lot the training of the GAN and, in my case, produced better results compared to WGAN.

## Results with CelebA Dataset
I ran experiments with different architectures for both the generator and the critic networks, varying the number of filters respectively in the last and the first layer.

Then I also tried to generate 128x128 images, instead of the standard 64x64 images I worked with in all the previous experiments.

### 128 filters for critic and generator - 64x64 images
First experiment for the WGAN-GP where I obtained the best results.

The critic and the generator were trained for 14 epochs and we can clearly see that both the negative loss of the critic and the Wasserstein distance estimate are steadily decreasing, indicating a very stable training.

As we can see, the both the critic loss and the Wasserstein distance are very significative, since a decrease means that the generated samples improved, as we can see in the gif reported after the charts.

<p align="center">
  <img hspace=20 src="WGAN-GP/results_19-03-19_13-59_e20_d128_g128_64x64/discriminator_loss_smoothed.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_13-59_e20_d128_g128_64x64/generator_loss_smoothed.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_13-59_e20_d128_g128_64x64/wasserstein_distance.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_13-59_e20_d128_g128_64x64/gradient_penalty.png" width="300" />
</p>
<p align="center">
  <img hspace=0 src="WGAN-GP/results_19-03-19_13-59_e20_d128_g128_64x64/video/celeba.gif" width="400" />
  <img hspace=0 src="WGAN-GP/results_19-03-19_13-59_e20_d128_g128_64x64/video/frame_13.png" width="400" /> 
</p>

The generated samples have a good quality and variety, indicating that no mode collapse happened.<br>
Probably by training the WGAN-GP for more epochs we could still improve the quality of the generated faces.

### 64 filters for critic and generator - 64x64 images
This is an esperiment were I tried to reduce the number of filters of both the critic and the generator networks.

The networks were trained for 16 epochs.

<p align="center">
  <img hspace=20 src="WGAN-GP/results_19-03-19_21-34_e20_d64_g64_64x64/discriminator_loss_smoothed.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_21-34_e20_d64_g64_64x64/generator_loss_smoothed.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_21-34_e20_d64_g64_64x64/wasserstein_distance.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_21-34_e20_d64_g64_64x64/gradient_penalty.png" width="300" />
</p>

Also in this experiment, like the previous one, both the critic loss and the Wasserstein distance are steadily decreasing, representing a good and costant learning. Also in this case, training for additional epochs could bring an improvement in the quality of the generated samples.

<p align="center">
  <img hspace=0 src="WGAN-GP/results_19-03-19_21-34_e20_d64_g64_64x64/video/celeba.gif" width="400" />
  <img hspace=0 src="WGAN-GP/results_19-03-19_21-34_e20_d64_g64_64x64/video/frame_15.png" width="400" /> 
</p>

We can see the quality of the faces increasing epoch by epoch, until the last one.<br>
The samples are not as good as the ones from the previous experiment with a double number of filters, but they are still quite good.

### 128 filters for critic and 64 for generator - 128x128 images
In this experiment I increased the image size from 64x64 to 128x128. In order to do this, I added an additional layer to the generator to double the size of its output.

Unfortunately the training times with the increased complexity of the GAN were very high, so I had to stop the training earlier after only 5 epochs.

<p align="center">
  <img hspace=20 src="WGAN-GP/results_19-03-19_18-41_e20_d128_g64_128x128/discriminator_loss_smoothed.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_18-41_e20_d128_g64_128x128/generator_loss_smoothed.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_18-41_e20_d128_g64_128x128/wasserstein_distance.png" width="300" />
  <img hspace=20 src="WGAN-GP/results_19-03-19_18-41_e20_d128_g64_128x128/gradient_penalty.png" width="300" />
</p>

As we can see, both the critic loss and the Wasserstein distance estimate are still decreasing at the end of the training, meaning that we would have obtained better results by training more.

<p align="center">
  <img hspace=0 src="WGAN-GP/results_19-03-19_18-41_e20_d128_g64_128x128/video/celeba.gif" width="400" />
  <img hspace=0 src="WGAN-GP/results_19-03-19_18-41_e20_d128_g64_128x128/video/frame_4.png" width="400" /> 
</p>

The generated samples of the interrupted training are quite promising, some faces are good while other ones are still missing some parts. This is also due to the increased complexity of the problem, with doubled dimensions of the images and more network weights to train.