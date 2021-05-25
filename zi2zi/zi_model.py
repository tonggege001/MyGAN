import torch
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, ngf=64, ch_in=1, ch_out=1):
        super().__init__()              # x = (X - kernel + 2 * padding) / stride + 1
        self.conv1 = nn.Conv2d(ch_in, ngf, kernel_size=4, stride=2, padding=1, bias=False)              # 64 x 64  -->  32 x 32
        # self.bn1 = nn.BatchNorm2d(ngf * 8)            # 第一层没有BN

        self.conv2 = nn.Conv2d(ngf, 2 * ngf, kernel_size=4, stride=2, padding=1, bias=False)        # 32 x 32  -->  16 x 16
        self.bn2 = nn.BatchNorm2d(2 * ngf)

        self.conv3 = nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=4, stride=2, padding=1, bias=False)    # 16 x 16  -->  8 x 8
        self.bn3 = nn.BatchNorm2d(4 * ngf)

        self.conv4 = nn.Conv2d(4 * ngf, 8 * ngf, kernel_size=4, stride=2, padding=1, bias=False)    # 8 x 8    -->  4 x 4
        self.bn4 = nn.BatchNorm2d(8 * ngf)

        self.conv5 = nn.Conv2d(8 * ngf, 8 * ngf, kernel_size=4, stride=2, padding=1, bias=False)    # 4 x 4    -->  2 x 2
        self.bn5 = nn.BatchNorm2d(8 * ngf)

        self.tconv5 = nn.ConvTranspose2d(8 * ngf, 8 * ngf, kernel_size=4, stride=2, padding=1, bias=False)      # 2 x 2  -->  4 x 4
        self.tbn5 = nn.BatchNorm2d(8 * ngf)

        self.tconv4 = nn.ConvTranspose2d(2 * 8 * ngf, 4 * ngf, kernel_size=4, stride=2, padding=1, bias=False)  # 4 x 4  -->  8 x 8
        self.tbn4 = nn.BatchNorm2d(4 * ngf)

        self.tconv3 = nn.ConvTranspose2d(2 * 4 * ngf, 2 * ngf, kernel_size=4, stride=2, padding=1, bias=False)  # 8 x 8  -->  16 x 16
        self.tbn3 = nn.BatchNorm2d(2 * ngf)

        self.tconv2 = nn.ConvTranspose2d(2 * 2 * ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False)      # 16 x 16 --> 32 x 32
        self.tbn2 = nn.BatchNorm2d(ngf)

        self.tconv1 = nn.ConvTranspose2d(2 * ngf, ch_out, kernel_size=4, stride=2, padding=1, bias=False)                # 32 x 32 -->  64 x 64

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x_train):         # x_train: batch_size x 1 x 64 x 64
        x_layer1 = self.leakyrelu(self.conv1(x_train))
        x_layer2 = self.leakyrelu(self.bn2(self.conv2(x_layer1)))
        x_layer3 = self.leakyrelu(self.bn3(self.conv3(x_layer2)))
        x_layer4 = self.leakyrelu(self.bn4(self.conv4(x_layer3)))
        x_layer5 = self.leakyrelu(self.bn5(self.conv5(x_layer4)))
        x_out5 = self.relu(self.tbn5(self.tconv5(x_layer5)))
        x_out4 = self.relu(self.dropout(self.tbn4(self.tconv4(torch.cat([x_layer4, x_out5], dim=1)))))
        x_out3 = self.relu(self.dropout(self.tbn3(self.tconv3(torch.cat([x_layer3, x_out4], dim=1)))))
        x_out2 = self.relu(self.dropout(self.tbn2(self.tconv2(torch.cat([x_layer2, x_out3], dim=1)))))
        x_out1 = self.tconv1(torch.cat([x_layer1, x_out2], dim=1))
        x_out = self.tanh(x_out1)
        return x_out


class Discriminator(nn.Module):
    def __init__(self, ndf=64, ch_in=1):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(2 * ch_in, ndf, kernel_size=5, stride=1, padding=2, bias=False)              # 64 x 64  -->  64 x 64
        self.conv2 = nn.Conv2d(ndf, 2 * ndf, kernel_size=4, stride=2, padding=1, bias=False)        # 64 x 64  -->  32 x 32
        self.bn2 = nn.BatchNorm2d(2 * ndf)

        self.conv3 = nn.Conv2d(2 * ndf ,4 * ndf, kernel_size=4, stride=2, padding=1, bias=False)    # 32 x 32  -->  16 x 16
        self.bn3 = nn.BatchNorm2d(4 * ndf)

        self.conv4 = nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=4, stride=2, padding=1, bias=False)    # 16 x 16  -->  8 x 8
        self.bn4 = nn.BatchNorm2d(8 * ndf)

        self.conv5 = nn.Conv2d(8 * ndf, 1, kernel_size=5, stride=1, padding=2, bias=False)          # 8 x 8  -->  8 x 8
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_edge, x_image):
        x_train = torch.cat([x_edge, x_image], dim=1)
        x_train = self.leakyrelu(self.conv1(x_train))
        x_train = self.leakyrelu(self.bn2(self.conv2(x_train)))
        x_train = self.leakyrelu(self.bn3(self.conv3(x_train)))
        x_train = self.leakyrelu(self.bn4(self.conv4(x_train)))
        x_train = self.sigmoid(self.conv5(x_train)).view(-1)
        return x_train











