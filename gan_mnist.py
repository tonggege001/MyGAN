import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64

'''
    用GAN生成MNIST
'''

def get_data():
    dataset = datasets.MNIST(root="./data/",train=True)
    x_train = (dataset.data.float().view(-1, 1, 28,28) - 127.5) / 127.5
    y_train = dataset.targets
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(128)#, momentum=0.9)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)#, momentum=0.9)
        self.batch_norm3 = torch.nn.BatchNorm2d(512)#, momentum=0.9)
        self.leakyrelu = nn.LeakyReLU()
        self.linear = nn.Linear(8192, 1)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batch_norm1(self.conv2(x)))
        x = self.leakyrelu(self.batch_norm2(self.conv3(x)))
        x = self.leakyrelu(self.batch_norm3(self.conv4(x)))
        x = torch.flatten(x).reshape(-1, 8192)
        x = torch.sigmoid(self.linear(x))
        return x

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.d = 3
        self.linear = nn.Linear(input_size, self.d*self.d*512)
        self.conv_tranpose1 = nn.ConvTranspose2d(512, 256, 5, 2, 1)
        self.conv_tranpose2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_tranpose3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv_tranpose4 = nn.ConvTranspose2d(64, 1, 3, 1, 1)
        self.batch_norm1 = torch.nn.BatchNorm2d(512)#, momentum=0.9)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)#, momentum=0.9)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)#, momentum=0.9)
        self.batch_norm4 = torch.nn.BatchNorm2d(64)#, momentum=0.9)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x).reshape(-1, 512, self.d, self.d)
        x = self.relu(self.batch_norm1(x))
        x = self.conv_tranpose1(x)
        x = self.relu(self.batch_norm2(x))
        x = self.conv_tranpose2(x)
        x = self.relu(self.batch_norm3(x))
        x = self.conv_tranpose3(x)
        x = self.relu(self.batch_norm4(x))
        x = self.tanh(self.conv_tranpose4(x))
        return x



def train():
    param = {
        "Epoch": 200,
    }

    train_loader = get_data()
    netG = Generator(100).to(device)
    netD = Discriminator(28).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=1e-4)
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    fake_label = torch.ones((batch_size, ), dtype=torch.float32).to(device)         # 这块进行了标签翻转
    real_label = torch.zeros((batch_size, ), dtype=torch.float32).to(device)
    for epoch in range(param["Epoch"]):
        netD.train()
        netG.train()
        r_correct = 0.0
        f_correct = 0.0
        total = 0.0
        print_gen_loss = 0.0
        print_dis_loss = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc=f"Epoch:{epoch}"):
            images = images.to(device)
            real_out = netD(images).view(-1)
            d_loss_real = criterion(real_out, real_label[:images.shape[0]])
            real_scores = real_out
            vec = torch.rand((batch_size,100)).to(device)
            fake_images = netG(vec)
            fake_out = netD(fake_images).view(-1)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out

            d_loss = d_loss_real + d_loss_fake
            optD.zero_grad()
            d_loss.backward()
            optD.step()
            print_dis_loss += d_loss
            # print("D batch Loss: {:.3}".format(d_loss))

            # Train Generator
            for j in range(30):                 # 生成器需要多学习一些，直到生成器生成的图片判别器识别率为0.5
                vec = torch.rand((batch_size,100)).to(device)
                fake_images = netG(vec)
                fake_out = netD(fake_images).view(-1)
                g_loss = criterion(fake_out, real_label)

                optG.zero_grad()
                g_loss.backward()
                optG.step()
                print_gen_loss += g_loss

                tmp_out = netD(fake_images).view(-1)
                tmp_scores = torch.tensor([1 if tmp_out[i] >= 0.5 else 0 for i in range(tmp_out.shape[0])], dtype=torch.long).to(device)
                t_correct = torch.sum(tmp_scores == fake_label[:tmp_scores.shape[0]])

                if t_correct/tmp_scores.shape[0] < 0.65:                    # 如果判别起分类正确过高说明生成器还需要继续学习
                    break

            total += real_scores.shape[0]
            total += fake_scores.shape[0]
            real_scores = torch.tensor([1 if real_scores[i] >= 0.5 else 0 for i in range(real_scores.shape[0])], dtype=torch.long).to(device)
            fake_scores = torch.tensor([1 if fake_scores[i] >= 0.5 else 0 for i in range(fake_scores.shape[0])], dtype=torch.long).to(device)
            r_correct += torch.sum(real_scores == real_label[:real_scores.shape[0]])
            f_correct += torch.sum(fake_scores == fake_label[:fake_scores.shape[0]])

        print("Training Discriminator Loss: {:.2}, Real Accuracy:{:.2}, Fake Accuracy: {:.2}, Generator Loss: {:.2}".
              format(print_dis_loss, 2 * r_correct / total, 2 * f_correct / total, print_gen_loss))

        if epoch % 10 == 0:
            with torch.no_grad():
                netD.eval()
                netG.eval()
                vec = torch.rand((5, 100)).to(device)
                fake_images = netG(vec).cpu()
                fake_images = torch.squeeze(fake_images, dim=1)
                for i in range(fake_images.shape[0]):
                    plt.axis("off")
                    plt.imshow(fake_images[i].reshape(28,28))
                    plt.savefig(f"tmp_images/Epoch {epoch}_{i}.png", bbox_inches='tight', dpi=300, pad_inches=0.0)
            torch.save(netG, "model/mnist_netG.pkl")
            torch.save(netD, "model/mnist_netD.pkl")




'''
训练GAN的一些心得体会
1. Discriminator尽量使用LeakyReLU, Generator可以使用ReLU
2. Generator的输出使用tanh, 那么训练集的数据要标准化到[-1, 1]
3. Generator和Discriminator的模型结构、优化器、学习率尽量保持一致
4. Generator训练较困难，所以可以先训练一批次D，然后训练3-5次G

5. 良好的训练过程应该是D和G的loss变化不大，并且D的real accuracy和fake accuracy接近0.5，也就是说不能够一开始就直接到了99%
'''










if __name__ == "__main__":
    train()








