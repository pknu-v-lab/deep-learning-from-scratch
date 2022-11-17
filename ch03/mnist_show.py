# coding: utf-8
import sys, os
sys.path.append(os.getcwd())  # 프로젝트 루트를 시스템 경로에 추가
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5

    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
    print(img.shape)  # (28, 28)

    img_show(img)
