import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math


class Discriminator(nn.Module):
    """ Defines a PatchGAN discriminator """

    def __init__(self, input_nc, embedding_num, ndf=64, norm_layer=nn.BatchNorm2d, image_size=256):
        """
        Construct a PatchGAN discriminator
        Parameter
        :param input_nc:        the number of channels in input images
        :param embedding_num:
        :param ndf:             the number of filters in the first conv layer
        :param norm_layer:      normalization layer
        :param image_size:
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:           # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

# As tf implementation, kernel_size = 5, use "SAME" padding, so we should use kw = 5 and padw = 2
        kw = 5
        padw = 2
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1,3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

        # final_channels = ndf * nf_mult
        final_channels = 1
        # use stride of 2 conv2 layer 3 times, cal the image_size
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)

        final_features = final_channels * image_size * image_size
        self.binary = nn.Linear(final_features, embedding_num)
        self.category = nn.Linear(final_features,embedding_num)

    def forward(self, x_train):
        """ Standard forward """
        features = self.model(x_train)
        features = features.view(x_train.shape[0], -1)
        binary_logits = self.binary(features)
        category_logits = self.category(features)
        return binary_logits, category_logits











