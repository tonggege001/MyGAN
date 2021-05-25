import io
import pygame
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def gen_one_word(font_idx, word):
    pygame.init()
    font = pygame.font.Font("fonts/{}.ttf".format(font_idx), 64)

    # 渲染图片，设置背景颜色和字体样式, 前面的颜色是字体颜色
    rtext = font.render(word, True, (255, 255, 255), (0, 0, 0))
    # 保存图片
    pygame.image.save(rtext, "image.jpg")  # 图片保存地址



def gen_words_images():
    word_dic = {}
    idx_dic = {}
    with open("word.txt", "r", encoding="utf-8") as f:
        item = f.readline().split(" ")
        for idx, word in enumerate(item):
            idx_dic[idx] = word
            word_dic[word] = idx

    pygame.init()
    font1 = pygame.font.Font("fonts/1.ttf", 64)
    font2 = pygame.font.Font("fonts/2.ttf", 64)
    font3 = pygame.font.Font("fonts/3.ttf", 64)
    font4 = pygame.font.Font("fonts/4.ttf", 64)
    for i in range(len(word_dic.keys())):
        rtext1 = font1.render(idx_dic[i], True, (255, 255, 255), (0, 0, 0))
        rtext2 = font2.render(idx_dic[i], True, (255, 255, 255), (0, 0, 0))
        rtext3 = font3.render(idx_dic[i], True, (255, 255, 255), (0, 0, 0))
        rtext4 = font4.render(idx_dic[i], True, (255, 255, 255), (0, 0, 0))
        pygame.image.save(rtext1, "data/1/{}.png".format(i))
        pygame.image.save(rtext2, "data/2/{}.png".format(i))
        pygame.image.save(rtext3, "data/3/{}.png".format(i))
        pygame.image.save(rtext4, "data/4/{}.png".format(i))


def trim_one_image(font_idx, idx):
    path = "data/{}/{}.png".format(font_idx, idx)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.array(image, dtype=np.long)
    arr = np.array(image, dtype=np.long)
    if(arr.sum()) == 0:
        return None, False

    vertical = arr.sum(axis=1)
    horizontal = arr.sum(axis=0)
    top, left, right, bottom = 0,0,0,0
    for i in range(horizontal.shape[0]):
        if horizontal[i] != 0:
            left = i
            break
    for i in range(horizontal.shape[0]-1, -1, -1):
        if horizontal[i] != 0:
            right = i
            break

    for i in range(vertical.shape[0]):
        if vertical[i] != 0:
            top = i
            break

    for i in range(vertical.shape[0]-1, -1, -1):
        if vertical[i] != 0:
            bottom = i
            break

    if bottom - top + 1 > 64:
        return None, False
    if right - left + 1 > 64:
        return None, False

    height = bottom - top + 1
    width = right - left + 1
    corner = {"left": int(32 - width / 2), "top": int(32 - height / 2)}

    arr = np.zeros((64, 64), dtype=np.long)
    for i in range(height):
        for j in range(width):
            arr[corner["top"]+i][corner["left"] + j] = image[top + i][left + j]
    return arr, True


def trim_images():
    err_idx = []
    font1 = []
    font2 = []
    font3 = []
    font4 = []
    for idx in range(3328):
        npy_list = []
        for font_idx in [1, 2, 3, 4]:
            word1, state1 = trim_one_image(font_idx, idx)
            if state1:
                npy_list.append(word1)
            else:
                continue
        if len(npy_list) != 4:              # 四个字体必须全部都修剪成功
            err_idx.append(idx)
            continue

        font1.append(npy_list[0])
        font2.append(npy_list[1])
        font3.append(npy_list[2])
        font4.append(npy_list[3])

    font1 = np.array(font1, dtype=np.int32)
    font2 = np.array(font2, dtype=np.int32)
    font3 = np.array(font3, dtype=np.int32)
    font4 = np.array(font4, dtype=np.int32)

    print("Trimmed Ok, err_idx: {}", err_idx)
    np.savez("data/font.npz", font1, font2, font3, font4)


def get_data(x_edge_idx, x_image_idx):
    data = np.load("data/font.npz")
    x_edge = data["arr_{}".format(x_edge_idx-1)].astype(np.float).reshape((-1, 1, 64, 64))
    x_train = data["arr_{}".format(x_image_idx - 1)].astype(np.float).reshape((-1, 1, 64, 64))

    x_edge = torch.tensor((x_edge - 127.5) / 127.5, dtype=torch.float32)
    x_train = torch.tensor((x_train - 127.5) / 127.5, dtype=torch.float32)
    return x_edge, x_train



if __name__ == "__main__":
    # gen_words_images()
    # trim_images()
    pass

