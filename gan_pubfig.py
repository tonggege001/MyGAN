import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 60


def get_data():
    # npz 已经归一化到(0, 1) 之间
    data = np.load("data/PubFig.npz")
    x_train, y_train, x_test, y_test = (data["arr_0"] * 255).astype(np.uint8), data["arr_1"], (data["arr_2"]*255).astype(np.uint8), data["arr_3"]
    x_train = np.concatenate((x_train, x_test), axis=0)
    new_train = np.zeros((x_train.shape[0], 256, 256, 3), dtype=np.float32)
    y_train = np.concatenate((y_train, y_test), axis=0)
    for i in range(x_train.shape[0]):
        image = Image.fromarray(x_train[i])
        image = image.resize((256, 256))
        new_train[i] = np.array(image)

    x_train = torch.tensor((new_train - 127.5) / 127.5, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=3)
    return dataloader


# 在netG和netD上调用的自定义权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, vec_len, ngf, out_channels):
        super(Generator, self).__init__()                   # x = (X - kernel + 2 * padding)/stride + 1
        self.main = nn.Sequential(
            # input: vec_len x 1 x 1, output: ngf * 8 x 4 x 4
            nn.ConvTranspose2d(vec_len, ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # input: ngf*16 x 4 x 4, output: ngf * 8 x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # input: ngf*8 x 8 x 8, output: ngf * 4 x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # input: ngf * 4 x 16 x 16, output: ngf * 2 x 32 x 32
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # input: ngf * 2 x 32 x 32, output: ngf x 64 x 64
            nn.ConvTranspose2d(ngf * 8,  ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # input: ngf * 2 x 64 x 64, output: ngf x 128 x 128
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # input: ngf * 2 x 128 x 128, output: ngf x 256 x 256
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # input: ngf x 32 x 32, output: out_channels x
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )


    def forward(self, x_train):
        return self.main(x_train)


class Discriminator(nn.Module):
    def __init__(self, ndf, input_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: input_channels x 256 x 256, output: ndf x 128 x 128
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input: ndf x 128 x 128, output: 2*ndf x 64 x 64
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),


            # input: 2*ndf x 64 x 64, output: 4*ndf x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # input: 4*ndf x 32 x 32, output: 8*ndf x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            # input: 4*ndf x 16 x 16, output: 8*ndf x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            # input: 4*ndf x 8 x 8, output: 8*ndf x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            # input: 4*ndf x 4 x 4, output: 8*ndf x 2 x 2
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2),

            # input: 8*ndf x 2 x 2
            nn.Conv2d(ndf * 16, 1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_train):
        return self.main(x_train)


def train():
    param = {
        "Epochs": 200000,
        "vec_len": 200
    }
    train_loader = get_data()
    netG = Generator(vec_len=param["vec_len"], ngf=64, out_channels=3).to(device)
    netG = torch.load("model/pubfig_netG.pkl", map_location=device)
    # netG.apply(weights_init)
    netD = Discriminator(ndf=64, input_channels=3).to(device)
    netD = torch.load("model/pubfig_netD.pkl", map_location=device)
    # netD.apply(weights_init)

    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=3e-4, betas=(0.5, 0.999))

    real_label = torch.ones((batch_size * 2,), dtype=torch.float32).to(device)            # 这块对D进行标签翻转，更容易训练
    fake_label = torch.zeros((batch_size * 2,), dtype=torch.float32).to(device)
    fix_vector = torch.rand((5, param["vec_len"], 1, 1)).to(device)

    for epoch in range(param["Epochs"]):
        netD.train()
        netG.train()
        log = {                             # 需要用到的统计值
            "d_fake_loss": 0.0,
            "d_real_loss": 0.0,
            "d_fake_correct": 0.0,
            "d_real_correct": 0.0,
            "d_loss": 0.0,
            "g_loss": 0.0,
            "b_correct": 0.0,
            "total": 0.0
        }
        for images, _ in tqdm.tqdm(train_loader, desc=f"Epoch: {epoch}"):
            # 训练Discriminator
            images = images.to(device)
            real_out = netD(images).view(-1)
            d_real_loss = criterion(real_out, real_label[:real_out.shape[0]])

            vec = torch.rand((images.shape[0], param["vec_len"], 1, 1)).to(device)
            fake_images = netG(vec)

            fake_out = netD(fake_images.detach()).view(-1)
            d_fake_loss = criterion(fake_out, fake_label[:fake_out.shape[0]])

            d_loss = d_real_loss + d_fake_loss
            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # 计算打印变量
            real_out = torch.tensor([1 if real_out[e] >= 0.5 else 0 for e in range(real_out.shape[0])], dtype=torch.long).to(device)
            fake_out = torch.tensor([1 if fake_out[e] >= 0.5 else 0 for e in range(fake_out.shape[0])], dtype=torch.long).to(device)
            log["d_real_correct"] += torch.sum(torch.eq(real_out, real_label[:real_out.shape[0]]))
            log["d_fake_correct"] += torch.sum(torch.eq(fake_out, fake_label[:fake_out.shape[0]]))
            log["total"] += (real_out.shape[0] + fake_out.shape[0])
            log["d_real_loss"] += d_real_loss
            log["d_fake_loss"] += d_fake_loss
            log["d_loss"] += d_loss

            vec = torch.rand((images.shape[0] * 2, param["vec_len"], 1, 1)).to(device)
            fake_images = netG(vec)
            fake_out = netD(fake_images).view(-1)
            g_loss = criterion(fake_out, real_label[:fake_out.shape[0]])

            optG.zero_grad()
            g_loss.backward()
            optG.step()

            # 计算要打印的变量
            log["g_loss"] += g_loss

        # 打印该轮的信息，保存相应的图片
        print("Discriminator Loss: {:.2f}, Real Accuracy: {:.2f}, Fake Accuracy: {:.2f}, Generator Loss: {:.2f}".format(
            log["d_loss"], 2. * log["d_real_correct"] / log["total"], 2. * log["d_fake_correct"] / log["total"],
            log["g_loss"]
        ))

        # 保存每轮的图片
        if epoch % 5 == 0:
            with torch.no_grad():
                netD.eval()
                netG.eval()
                fake_images = netG(fix_vector).cpu().permute(0, 2, 3, 1)
                for i in range(fake_images.shape[0]):
                    plt.axis("off")
                    plt.imshow((fake_images[i] + 1)/2)
                    plt.savefig(f"images/Epoch {epoch}_{i}.png", bbox_inches='tight', dpi=300, pad_inches=0.0)
            torch.save(netG, "model/pubfig_netG.pkl")
            torch.save(netD, "model/pubfig_netD.pkl")


if __name__ == "__main__":
    train()

