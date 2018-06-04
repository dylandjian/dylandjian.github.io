---
title: World Models applied to Sonic
date: "2018-05-31T22:12:03.284Z"
---

Open AI lauched a Reinforcement Learning competition called the [Retro Contest](https://contest.openai.com/) on April 5th. The contest lasted over the course of the past 2 months, and the goal was to make the best agent to play the old SEGA Genesis platform game Sonic the Hedgehog. The problem is really simple to understand, yet very **very** hard to resolve. The evaluation of the agent is done on levels that have not been previously seen by the agent, therefore making the competition a Meta Reinforcement Learning problem. I will present how I tried to tackle this problem by applying recently published techniques that have been applied to similar problems (however generally simpler). This article will require a bit of background knowledge in Machine Learning and Python, as I will be referencing [my own implementation](https://github.com/dylandjian/retro-contest-sonic).

## Introduction

I started the contest about 3 weeks ago, with general knowledge about Machine Learning and Deep Learning as a self thought practitioner and student in software development. My only other experience with a large Reinforcement Learning problem was implementing AlphaGo Zero from scratch, using (mainly) PyTorch. [My article on the subject (coming soon)](https://dylandjian.github.io/alphago-zero/) and [my implementation on Github](https://github.com/dylandjian/superGo).
I followed the [guidelines](https://contest.openai.com/details) to get started and submitted my first agent using one of the baselines algorithm (JERK: _Just Enough Retained Knowledge_).
When it was time to start thinking about a way to formulate a good answer to the problem, a few ideas came to my mind : PPO, DQN and it's variations or perhaps TRPO. However, these algorithm have already proven their worth and are known performers. I wanted to try something different even though it might not give any successful results. I had read the paper about **World Models** a few weeks prior to starting the contest. I had thought about a similar approach before reading the paper, but never actually took the time to experiment about it. I figured it was the perfect time to apply this really interesting approach to a concrete problem.

## World Models

The algorithm is divided in 3 main components that have their own logic : **V**isual, **M**emory, **C**ontroller. The idea behind it is pretty elegent : As humans, we learn our own abstract representation of the dynamics of the world we live in, whether it is in space or in time. We have the ability to _roughly_ visualize a concept when we think about one. Let's say I ask you to visualize what a Sonic level might look like to you. If you have already seen or played Sonic, you are probably thinking about a rough sketch of a level, not the RBG value of every pixel on the screen. If I now ask you to imagine how Sonic is going to evolve on that level you just thought of, you can probably see him move through the level as well avoiding cracks, enemy units and getting yellow rings in order to achieve a higher score.
The goal of the paper was to replicate this beautiful architecture, which they successfully did on 2 environments : the _CarRacing-v0_ for OpenAI Gym, and Doom. My goal was to try to apply this architecture it to Sonic.

### Visual Model

### Concept

The visual model that is supposed to create the abstract representation of the space is called an Autoencoder. It is basically made of 2 components, an encoder and a decoder. The job of the encoder is to compress the observation (in our case the RGB frame of the game) into a vector with a much smaller dimension (generally between 10 and 300 dimensions, 64 in the paper for Doom). On the other hand, the job of the decoder is to try to recreate the original RGB frames from the compressed vector. The Autoencoder variant that has been used in the paper is called a Variational Autoencoder (_VAE_). [Here is a good resource on the subject](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/). Instead of encoding the frames to a latent variable _z_, the encoder tries to compress the frame into a Normal probability distribution with mean 0 and standard deviation of 1.
Also, since we are dealing with images, we use convolutions in order to capture local information instead of using pixel values directly.
I chose to implement β-VAE to get a more robust representation of the Sonic levels.

### Code

Let's take a look at a possible implementation. The architecture follows the one proposed on the paper, except that it has one more layer because the frames taken from the game have been resized to 128x128 (using the "nearest" method) instead of 64x64 in the paper.

```
class ConvVAE(nn.Module):
    def __init__(self, input_shape, z_dim):
        super(ConvVAE, self).__init__()

        ## Encoder
        self.image_size = 3 * WIDTH * HEIGHT
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        ## Latent representation
        self.fc1 = nn.Linear(256 * 6 * 6, z_dim)
        self.fc2 = nn.Linear(256 * 6 * 6, z_dim)
        self.fc3 = nn.Linear(z_dim, 256 * 6 * 6)

        ## Decoder
        self.deconv1 = nn.ConvTranspose2d(256 * 6 * 6, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 6, stride=2)
```

First, we define our layers. 5 convolutions that are mapped onto 2 linear vectors representing the mean and the standard deviation of our VAE. Then we add another linear layer that takes the ouput _(mean, std)_ and map it to a vector that will be the input of the decoder. The decoder will reconstruct an image that has the size of the input image in order to calculate the loss function.

Now onto the forward pass.

```
def encode(self, x):
   h = F.relu(self.conv1(x))
   h = F.relu(self.conv2(h))
   h = F.relu(self.conv3(h))
   h = F.relu(self.conv4(h))
   h = h.view(-1, 256 * 6 * 6)
   return self.fc1(h), self.fc2(h)

def reparameterize(self, mu, logvar):
   std = torch.exp(0.5 * logvar)
   eps = torch.randn_like(std)
   return eps * std + mu

def decode(self, z):
   h = self.fc3(z).view(-1, 256 * 6 * 6, 1, 1)
   h = F.relu(self.deconv1(h))
   h = F.relu(self.deconv2(h))
   h = F.relu(self.deconv3(h))
   h = F.relu(self.deconv4(h))
   h = F.sigmoid(self.deconv5(h))
   return h

def forward(self, x, encode=False):
   mu, logvar = self.encode(x)
   if encode:
       return mu
   z = self.reparameterize(mu, logvar)
   return self.decode(z), mu, logvar
```

The encode and decode methods do as their name suggest. The reparameterize method does sample from the mean and std during training, and returns the mean when infering. However, to make sure that our M model doesn't overfit to specific latent representation while learning how to model the time dependency on the space, we make sure to sample from these 2 vectors, even during inference time.

The final piece is the loss function :

```
def loss_fn(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss /= batch_size
    kld /= batch_size
    return loss + BETA * kld.sum()
```

The loss function for the β-VAE is defined as follows :  
  
$E_{q_\phi(z|x)}[\log p_θ(x|z)] - βD_{KL}(q_{\phi}(z|x)\ ||\ p(z))$
  
The left term is the marginal likelihood which measures how close the predicted frame was from the original frame, and the right term in the Kullback–Leibler divergence (or relative entropy) which is a measure of how the predicted frame diverges from the original frame when considered as a probability distribution, under the assumption that $p(z)$ and $q_{\phi}(z|x)$ are parametrised as Gaussians distributions.  
I chose a β value of 4 in most of my experiments to enforce a better latent representation, despite the potential quality loss on the overall reconstructed image. I also normalized each component of the loss by the number of example in the batch to get a more representative value.

### Results



### Memory Model

### Controller Model

## References

* [World Models](https://arxiv.org/pdf/1803.10122.pdf) - David Ha & Jürgen Schmidhuber
* [β-VAE](https://arxiv.org/pdf/1804.03599.pdf) - DeepMind
* [A tutorial on Mixed Density Networks](https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb) - David Ha
