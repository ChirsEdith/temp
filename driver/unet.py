# -*- coding:utf-8 -*-
# author: 我
#from keras import models
import os

from keras.utils.data_utils import Sequence

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
#from keras import layers, models
from tensorflow.keras import layers, models

from core import locate_and_correct

from math import ceil,floor

def generate_arrays_from_file(image,batch_size):
    count = 1
    steps = floor(len(image)/batch_size)
    while 1:
        if count == 1:
            np.random.shuffle(image)
        batch = image[(count - 1) * batch_size:count * batch_size]

        img = [cv2.imread('train_image/%s' % image) for image in batch]
        label = [cv2.imread('train_label/%s' % image) for image in batch]
        batch_x = np.array(img).astype(np.float32)
        batch_y = np.array(label).astype(np.float32)
        count = count % steps +1

        yield (batch_x, batch_y)

#steps_per_epoch 每执行一次steps,就去执行一次生产函数generate_arrays_from_file
# max_queue_size 从生产函数中出来的数据时可以缓存在queue队列中

def unet_train():
    height = 480
    width = 640
    input_name = os.listdir('train_image') # 原图像

    image_names= []
    for root, dirs, files in os.walk('train_image'):
        for image in files:
            print("正在读取%s" % image)
            #img = cv2.imread('train_image/%s' % image)
            #label = cv2.imread('train_label/%s' % image)
            image_names.append(image)
            #y_train.append(label)

    #X_train = np.array(X_train)
    #y_train = np.array(y_train)

    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    inpt = layers.Input(shape=(height, width, 3))
    conv1 = Conv2d_BN(inpt, 8, (3, 3)) #每两个3*3的卷积核后面跟一个2*2的池化
    conv1 = Conv2d_BN(conv1, 8, (3, 3))
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    conv2 = Conv2d_BN(conv2, 16, (3, 3))
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    conv3 = Conv2d_BN(conv3, 32, (3, 3))
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    # 减少过拟合
    conv5 = layers.Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    concat1 = layers.concatenate([conv4, convt1], axis=3)
    concat1 = layers.Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 64, (3, 3))
    conv6 = Conv2d_BN(conv6, 64, (3, 3))

    convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    concat2 = layers.concatenate([conv3, convt2], axis=3)
    concat2 = layers.Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 32, (3, 3))
    conv7 = Conv2d_BN(conv7, 32, (3, 3))

    convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    concat3 = layers.concatenate([conv2, convt3], axis=3)
    concat3 = layers.Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16, (3, 3))
    conv8 = Conv2d_BN(conv8, 16, (3, 3))

    convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    concat4 = layers.concatenate([conv1, convt4], axis=3)
    concat4 = layers.Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8, (3, 3))
    conv9 = Conv2d_BN(conv9, 8, (3, 3))
    conv9 = layers.Dropout(0.5)(conv9)
    outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

    model = models.Model(inpt, outpt)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()


    print("开始训练u-net")

    #print(X_train.shape, y_train.shape)
    #generator= MnistSequence(X_train, y_train, 32)
    size=16
    model.fit_generator(generate_arrays_from_file(image_names, batch_size=size), steps_per_epoch=floor(len(image_names)/size) ,  epochs=10)
    #model.fit(X_train.astype('float32'), y_train.astype('float32'), epochs=10, batch_size=10)#epochs和batch_size看个人情况调整，batch_size不要过大，否则内存容易溢出

    model.save('unet.h5')
    print('unet.h5保存成功!!!')


def unet_predict(unet, img_src_path):
    img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
    # img_src=cv2.imread(img_src_path)
    #print(img_src.shape)
    if img_src.shape != (480, 640, 3):
        img_src = cv2.resize(img_src, dsize=(480, 640), interpolation=cv2.INTER_AREA)[:, :, :3]  # dsize=(宽度,高度),[:,:,:3]是防止图片为4通道图片，后续无法reshape
    img_src = img_src.reshape(1, 480, 640, 3)  # 预测图片shape为(1,512,512,3)

    img_mask = unet.predict(img_src)  # 归一化除以255后进行预测

    img_src = img_src.reshape(480, 640, 3)  # 将原图reshape为3维
    img_mask = img_mask.reshape(480, 640, 3)  # 将预测后图片reshape为3维

    img_mask = img_mask / np.max(img_mask) * 255  # 归一化后乘以255)

    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # 三个通道保持相同
    img_mask = img_mask.astype(np.uint8)  # 将img_mask类型转为int型
    #print(img_mask)
    cv2.imshow('mask', img_mask)
    cv2.waitKey(0)

    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    #print(img_mask)
    _, img_mask = cv2.threshold(img_mask, 15, 255, cv2.THRESH_BINARY)
    return img_src, img_mask


if __name__ == '__main__':
    unet_train()
    kernel = np.ones((3, 3), np.uint8)
    unet = models.load_model('unet.h5')
    #count=1
    for i in range(1, 1500):
        img_src_path = 'images/'+str(i)+'.JPG'
        img_src, img_mask = unet_predict(unet, img_src_path)

        cv2.imshow('mask', img_mask)
        cv2.waitKey(0)

        img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # 利用cor

        #cv2.imshow('src', img_src)
        #cv2.waitKey(0)

        #img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
        # 画了框的图片
        cv2.imshow('mask', img_src_copy)
        cv2.waitKey(0)

        # 每个数字
        #for img in Lic_img:
        #    cv2.imshow('lic', img)
        #    cv2.waitKey(0)


        """
        将数字抠出来存下，之后要手动打标注，图片名为no_value，no的图片的序号，
        value是值，0-9
        """
        #for img in Lic_img:
            #cv2.imwrite('num_images/%s.JPG'%str(count),img)
            #count = count+1
        #cv2.imshow('Lic', img)
        #cv2.waitKey(0)
