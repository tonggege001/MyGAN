import cv2
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import torch

def make_edges(x_train):
    # input: (batch_size, W, H, C), 255
    res = np.zeros(x_train.shape, dtype=np.uint8)
    for i in range(x_train.shape[0]):
        img = x_train[i].copy()
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gaussImg = cv2.Canny(blurred, 10, 70)
        gaussImg = 255 - gaussImg
        gaussImg = np.transpose(np.tile([gaussImg], (3, 1, 1)), (1, 2, 0))
        res[i] = gaussImg
    return res


def data_augmentation():
    datagen = ImageDataGenerator(rotation_range=180, width_shift_range=0.3, height_shift_range=0.3,
                                 shear_range=0.3, zoom_range=0.5, fill_mode="constant", cval=255,
                                 horizontal_flip=True, vertical_flip=True)
    x_edges = np.zeros((30*1000, 256, 256, 3), dtype=np.uint8)
    x_images = np.zeros((30*1000, 256, 256, 3), dtype=np.uint8)

    idx = 0
    for name in range(1, 31):
        file_name = "data/org/{}.png".format(name)
        x = cv2.imread(file_name, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (256, 256))
        x = np.array(x, dtype=np.uint8).reshape((1,) + x.shape)

        for batch in datagen.flow(x, batch_size=1):
            batch = batch.astype(np.uint8)
            edges = make_edges(batch)
            x_edges[idx: idx+1] = edges
            x_images[idx:idx+1] = batch
            idx += 1
            if idx % 1000 == 0:
                break
    np.savez("edge2dog.npz", x_edges, x_images)

def get_externel_edge():
    x_edges = np.zeros((3, 256, 256, 3), dtype=np.uint8)
    x = cv2.imread("data/externel/1.png", cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (256, 256))
    x = np.array(x, dtype=np.uint8).reshape((1,) + x.shape)
    edges = make_edges(x)
    x_edges[0:1] = edges

    x = cv2.imread("data/externel/4.jpg", cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (256, 256))
    x = np.array(x, dtype=np.uint8).reshape((1,) + x.shape)
    x_edges[1:2] = x

    x = cv2.imread("data/externel/5.jpg", cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (256, 256))
    x = np.array(x, dtype=np.uint8).reshape((1,) + x.shape)
    x_edges[2:3] = x

    np.savez("edge2dog_sample.npz", x_edges)


def get_data():
    data = np.load("edge2dog.npz")
    x_edge = data["arr_0"].astype(np.float)
    x_edge = np.transpose(x_edge, (0, 3, 1, 2))
    x_train = data["arr_1"].astype(np.float)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    x_edge = torch.tensor((x_edge - 127.5) / 127.5, dtype=torch.float32)
    x_train = torch.tensor((x_train - 127.5) / 127.5, dtype=torch.float32)

    data_sample = np.load("edge2dog_sample.npz")
    x_edge_sample = data_sample["arr_0"].astype(np.float)
    x_edge_sample = np.transpose(x_edge_sample, (0, 3, 1, 2))
    x_edge_sample = torch.tensor((x_edge_sample - 127.5) / 127.5, dtype=torch.float32)

    return x_edge, x_train, x_edge_sample



if __name__ == "__main__":
    # data_augmentation()
    # get_externel_edge()
    pass

