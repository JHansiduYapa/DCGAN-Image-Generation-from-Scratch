# Deep Convolutional Generative Adversarial Network (DCGAN)

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic images using the MNIST dataset.

## 1. Understanding Transpose Convolution
**Transpose convolution** is a crucial operation used in the generator of DCGANs to upsample feature maps. It works by reversing the convolution operation, allowing for the creation of larger outputs from smaller inputs.

### Visual Representation of Transpose Convolution
![Transpose Convolution](https://github.com/janith99hansidu/DCGAN-PyTorch/blob/main/src/transpose_convolution.png)

## 2. Explanation of Loss Functions
In DCGAN, we have two main components:
- **Generator Loss**: Measures how well the generator fools the discriminator.
- **Discriminator Loss**: Measures how well the discriminator distinguishes between real and generated images.

These losses help improve both components iteratively.

### How the Loss Functions Work
The generator’s goal is to minimize `log(1 - D(G(z)))`, but to improve training stability, this is often modified to maximizing `log(D(G(z)))`. The discriminator aims to maximize `log(D(x)) + log(1 - D(G(z)))`.

## 3. MNIST Dataset for Experiments
The network was trained using the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits. Below are example results showing how the loss changes over time:

### Training Loss Graphs
![Training Loss Graphs](https://github.com/janith99hansidu/DCGAN-PyTorch/blob/main/src/loss.png)

## 4. Series of Training Results
Below is a GIF showing the progression of generated images as training progresses. It helps visualize how the generator improves and produces increasingly realistic images.

### GIF of Training Progress
![Training Progress](https://github.com/janith99hansidu/DCGAN-PyTorch/blob/main/src/learning.gif)

## 5. Final Results: Real vs. Generated Images
Here’s a comparison between real MNIST images and the final outputs from the trained DCGAN model.

### Side-by-Side Comparison
![Real vs Generated Images](https://github.com/janith99hansidu/DCGAN-PyTorch/blob/main/src/real_vs_fake.png)

## References

1. [Generative Adversarial Networks (Original Paper)](https://arxiv.org/abs/1406.2661)
2. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
3. [DCGAN Tutorial by PyTorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
4. [PyTorch-GAN Repository by Erik Lindernoren](https://github.com/eriklindernoren/PyTorch-GAN)
5. [Vanilla GAN PyTorch Notebook by gordicaleksa](https://github.com/gordicaleksa/pytorch-GANs/blob/master/Vanilla%20GAN%20(PyTorch).ipynb)

## Conclusion

This project demonstrates how DCGANs can effectively generate realistic images from simple noise vectors. By iteratively training the generator and the discriminator, the network learns to produce images that resemble real data.
