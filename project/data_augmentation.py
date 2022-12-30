import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#生成延伸資料集-針對原圖縮放並移動成為新的資料

datagen = ImageDataGenerator(
        rescale=0.9,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest')

file_list = [file for file in os.listdir('dataset/train') if '_Augmentation' not in file]
for file in file_list:
    if 'Bold' in file:
        ASCII = 65 if 'Capital' in file else 97
        for i in range(ASCII, ASCII+26):
            try:
                letter = chr(i)
                datas = os.listdir('dataset/train/{0}/{1}'.format(file, letter))
                save_dir = 'dataset/train/{0}_Augmentation/{1}'.format(file, letter)
                os.mkdir(save_dir)
                for index, data in enumerate(datas):
                    list = []
                    path = 'dataset/train/{0}/{1}/{2}'.format(file, letter, data)
                    img = cv2.imread(path, 0)
                    size = random.randint(80, 85)
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                    pad1 = round((100 - size)/2)
                    pad2 = 100 - size - pad1
                    img = np.pad(img, ((pad1, pad2), (pad1, pad2)), 'constant', constant_values=(255, 255))
                    list.append(tf.reshape(img, (100, 100, 1)))
                    img = np.array(list, dtype=float)
                    i = 0
                    for batch in datagen.flow(img, batch_size=1, save_to_dir=save_dir,
                                                  save_prefix=data.split('.')[0], save_format='png'):
                        icon = "|/-\\"
                        print(f'\r{letter} : {icon[index % 4]} {math.ceil((index+1) * 100 / len(datas))}% (data:{(index+1)}/{len(datas)})', end='')
                        i += 1
                        if i > 9:
                            break
            except:
                print('\n'+file+'_Augmentation already exist.')

