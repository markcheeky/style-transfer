# Neural Style Transfer
This repo contains a functional PyTorch implementation of neural style transfer (NST). It is inspired by David Foster's book *[Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/)* and [this PyTorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).


## What is NST?

NST is a CNN-based technique that repaints a given image using a certain artistic style specified by a reference picture. It was introduced by Gatys et al. in a paper called *Image Style Transfer Using Convolutional Neural Networks*.


## How much training data does NST need?

None. NST only uses one source image for content and one for style. It still needs a trained CNN, but we can use a general pretrained image recognition model like *vgg19* and perform style transfer without any additional training.


## How does it work?

We initialize the CNN's input with the original content image and minimize the NST loss with gradient descent. However, we modify the **network input, not the weights!** After enough optimization steps, the input of the network becomes a stylized image.

The NST loss consists of two components: content loss and style loss. The content loss makes sure we stay close to the original content, while the style loss ensures a style match. For a detailed explanation, see the book or the paper referenced above.
