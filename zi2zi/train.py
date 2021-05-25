import torch
import torch.nn as nn
import tqdm
from zi_model import Generator, Discriminator
import matplotlib.pyplot as plt
from data import get_data

batch_size = 60
Epochs = 200
L1_penalty = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, edge_samples, image_samples):
    # edge_samples = edge_samples.to(device)
    netG = Generator(ngf=128, ch_in=1, ch_out=1).to(device)
    netD = Discriminator(ndf=128, ch_in=1).to(device)
    optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss().to(device)
    reg_crt = nn.L1Loss().to(device)

    for epoch in range(Epochs):
        log = train_one_epoch(epoch, train_loader, criterion, reg_crt, netD, optD, netG, optG)

        print("Epoch: {}, Discriminator Loss: {:.3f}, Real Accuracy: {:.3f}, Fake Accuracy: {:.3f}".format(epoch+1, log["d_loss"]/log["total"],
              log["real_correct"], log["fake_correct"]))
        print("Generator Org Loss: {:.3f}, Generator Reg Loss: {:.3f}".format(log["g_org_loss"]/log["total"],
                                                                              log["g_reg_loss"]/log["total"]))

        # Save Model and Save Sample Images
        if epoch % 2 == 0:
            torch.save(netG, "./model/zi2zi_netG.pkl")
            torch.save(netD, "./model/zi2zi_netD.pkl")
            with torch.no_grad():
                fake_images = netG(edge_samples).detach().cpu().numpy()
                plt.figure()
                for i in range(fake_images.shape[0]):               # 60个
                    plt.subplot(8, 8, i+1)
                    plt.imshow((fake_images[i][0] + 1)/2, cmap="gray")
                    plt.xticks([])
                    plt.yticks([])
                plt.savefig("./images/epoch_{}.png".format(epoch), bbox_inches='tight', dpi=300, pad_inches=0.0)


def train_one_epoch(epoch, train_loader, criterion, reg_crt, netD, optD, netG, optG):
    log = {
        "d_loss":        0.0,
        "fake_correct":    0.0,
        "real_correct":    0.0,
        "total":            0.0,
        "g_org_loss":       0.0,
        "g_reg_loss":       0.0
    }

    real_label = torch.ones((batch_size * 64, ), dtype=torch.float32).to(device)
    fake_label = torch.zeros((batch_size * 64, ), dtype=torch.float32).to(device)
    for x_edge, x_images in tqdm.tqdm(train_loader, desc=f"Epoch: {epoch}"):
        # ----------   Training Discriminator   -----------
        x_edge = x_edge.to(device)
        x_images = x_images.to(device)

        real_out = netD(x_edge, x_images)
        real_loss = criterion(real_out, real_label)

        fake_images = netG(x_edge).detach()
        fake_out = netD(x_edge, fake_images)
        fake_loss = criterion(fake_out, fake_label)

        d_loss = (fake_loss + real_loss) / 2
        optD.zero_grad()
        d_loss.backward()
        optD.step()

        # log some details
        log["d_loss"] += d_loss
        log["fake_correct"] = fake_out.mean()
        log["real_correct"] = real_out.mean()
        log["total"] += x_images.shape[0]


        # ----------   Training Generator    ----------
        fake_images = netG(x_edge)
        fake_out = netD(x_edge, fake_images)
        loss_org = criterion(fake_out, real_label[:fake_out.shape[0]])
        loss_reg = reg_crt(fake_images, x_images)
        g_loss = loss_org + L1_penalty * loss_reg
        optG.zero_grad()
        g_loss.backward()
        optG.step()

        log["g_org_loss"] += loss_org
        log["g_reg_loss"] += loss_reg

    return log


def prepare_train_loader():
    x_edge, x_train = get_data(1, 4)
    dataset = torch.utils.data.TensorDataset(x_edge, x_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)
    return train_loader



if __name__ == "__main__":
    train_loader = prepare_train_loader()
    print("Prepared Training Loader")
    x_edge = None
    x_images = None
    for edge, image in train_loader:
        x_edge = edge.clone().detach()
        x_images = image.clone().detach()
        break
    print("Begin Training......")
    plt.figure()
    for i in range(x_edge.shape[0]):  # 60个
        plt.subplot(8, 8, i + 1)
        plt.imshow((x_edge[i][0] + 1) / 2, cmap="gray")
    plt.savefig("./images/x_edge.png", bbox_inches='tight', dpi=300, pad_inches=0.0)

    train(train_loader, x_edge, x_images)


