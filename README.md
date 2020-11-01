  - > ***CNN ARCHITECTURES -***

**<u>LeNet</u>**

**<u>LeNet is a 7-level convolutional network by LeCun in 1998
</u>**that classifies digits and used by several banks to recognise
hand-written numbers on cheques digitized in **<u>32x32 pixel greyscale
input images</u>**. The ability to process higher resolution images
requires larger and more convolutional layers, so this technique is
constrained by the availability of computing resources.

![](media/image1.png)

**<u>AlexNet</u>**

  - > It consists of 5 convolutional(CONV) layers and 3 FC layers. And
    > here ReLU is used as activation function.

  - > It accepts input of size 227x227 of 3 channels (i.e 227x227x3)

  - > Gives output of size 1000x1

  - > More number of trainable parameters

It is much better than LeNet since,

  - > It supports RGB (LeNet supports only grayscale images)

  - > Input size of LeNet is 32x32 which is very small

> Etc.
> 
> ![](media/image4.png)

**<u>VGG (Visual Geometry Group)</u>**

(2014)

While previous derivatives of AlexNet focused on smaller window sizes
and strides in the first convolutional layer, VGG addresses another very
important aspect of CNNs: depth.

**<u>The architecture of VGG:</u>**

**Input:** VGG takes in a 224x224 pixel RGB image. For the ImageNet
competition, the authors cropped out the center 224x224 patch in each
image to keep the input image size consistent. Convolutional Layers.

**The convolutional layers:** in VGG use a very small receptive field
(3x3, the smallest possible size that still captures left/right and
up/down). There are also 1x1 convolution filters which act as a linear
transformation of the input, which is followed by a ReLU unit. The
convolution stride is fixed to 1 pixel so that the spatial resolution is
preserved after convolution.

**Fully-Connected Layers:** VGG has three fully-connected layers: the
first two have 4096 channels each and the third has 1000 channels, 1 for
each class.

**Hidden Layers:** All of VGG’s hidden layers use ReLU (a huge
innovation from AlexNet that cut training time). VGG does not generally
use Local Response Normalization (LRN), as LRN increases memory
consumption and training time with no particular increase in accuracy.

![](media/image3.png)

Ex: VGG16

![](media/image5.png)

1.  > Convolution using 64 filters

2.  > Convolution using 64 filters + Max pooling

3.  > Convolution using 128 filters

4.  > Convolution using 128 filters + Max pooling

5.  > Convolution using 256 filters

6.  > Convolution using 256 filters

7.  > Convolution using 256 filters + Max pooling

8.  > Convolution using 512 filters

9.  > Convolution using 512 filters

10. > Convolution using 512 filters + Max pooling

11. > Convolution using 512 filters

12. > Convolution using 512 filters

13. > Convolution using 512 filters + Max pooling

14. > Fully connected with 4096 nodes

15. > Fully connected with 4096 nodes

16. > Output layer with Softmax activation with 1000 nodes

**<u>ResNet (Visual Geometry Group)</u>**

(2015)

As per what we have seen so far, increasing the depth should increase
the accuracy of the network, as long as over-fitting is taken care of.
But the problem with increased depth is that the signal required to
change the weights, which arises from the end of the network by
comparing ground-truth and prediction becomes very small at the earlier
layers, because of increased depth. It essentially means that earlier
layers are almost negligible learned. This is called **vanishing
gradient.**

**<u>Residual Block:</u>**

In order to solve the problem of the vanishing/exploding gradient, this
architecture introduced the concept called Residual Network. In this
network we use a technique called skip connections . The skip connection
skips training from a few layers and connects directly to the output.
The approach behind this network is instead of layers learn the
underlying mapping, we allow network fit the residual mapping. So,
instead of say H(x), initial mapping, let the network fit, F(x) := H(x)
– x which gives H(x) := F(x) + x.

![](media/image6.png)

The advantage of adding this type of skip connection is because if any
layer hurt the performance of architecture then it will be skipped by
regularization. So, this results in training very deep neural network
without the problems caused by vanishing/exploding gradient.

![](media/image2.png)

**<u>Inception Net (GoogLeNet)</u>**

  -
