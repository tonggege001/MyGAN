import torch
import torch.nn as nn
import functools
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetGenerator, self).__init__()
        """
        Construct a Unet generator
        Parameters:
            input_nc:       the number of channels in input images
            output_nc:      the number of channels in output images
            num_downs:      the number of downsamplings in UNet. For example, if num_downs == 7, 
                                images of size 128 x 128 will become of size of size 1 x 1 at the bottleneck
            ngf:            the number of filters in the last conv layer
            norm_layer:     normalization layer
        We construct the Unet from innermost layer to outermost layer. It is a recursive process
        """
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, embedding_dim=embedding_dim)
        for _ in range(num_downs-5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def forward(self, x_train, style_or_label=None):
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            return self.model(x_train, self.embedder(style_or_label))
        else:
            return self.model(x_train, style_or_label)


class UNetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 embedding_dim=128, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        """
        Construct a UNet submodule with skip connections.
        Parameter:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True),
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv],
            up = [uprelu, upconv, nn.Tanh()]

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)


    def forward(self, x_train, style=None):
        if self.innermost:
            encode = self.down(x_train)
            if style is None:
                return encode
            enc = torch.cat([style.view(style.shape[0], style.shape[1], 1, 1), encode], dim=1)
            dec = self.up(enc)
            return torch.cat([x_train, dec], dim=1), enc.view(x_train.shape[0], -1)
        elif self.outermost:
            enc = self.down(x_train)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            return dec, encode

        else:       # add skip connection
            enc = self.down(x_train)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            return torch.cat([x_train, dec], dim=1), encode














