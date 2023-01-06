import cv2
import math
import time
import numpy as np
from dictionary import *

# 將資料和標籤轉換為numpy陣列
def data_append(file, file_name, X_train, Y_train, kind, type='one'):
    start = time.time()
    X_tmp, Y_tmp = [], []
    # 讀取圖片
    for index, data in enumerate(file_name):
        l = data.split('b)')[-1].split('-')[0] if '(b)' in data else data.split('-')[0]
        if type == 'all':
            if '_0_' in data:
                if '(b)' in data:
                    file = 'Bold_Capital_Augmentation' if ord(l) < 92 else 'Bold_Lowercase_Augmentation'
                else:
                    file = 'Normal_Capital_Augmentation' if ord(l) < 92 else 'Normal_Lowercase_Augmentation'
            else:
                if '(b)' in data:
                    file = 'Bold_Capital' if ord(l) < 92 else 'Bold_Lowercase'
                else:
                    file = 'Normal_Capital' if ord(l) < 92 else 'Normal_Lowercase'
        path = './dataset/{0}/{1}/{2}/{3}'.format(kind, file, l, data)
        img = cv2.imread(path, 0)
        X_tmp.append(img)

        if type == 'all':
            if ord(l) < 92:
                Y_tmp.append(ord(l) - 65)
            else:
                Y_tmp.append(ord(l) - 71)
        else:
            number = int(data.split('-')[-1].split('_0_')[0]) if 'Augmentation' in file \
                else int(data.split('-')[-1].split('.')[0])
            Y_tmp.append(number2category[number])

        print(
            f'\rLoading data : {l} - {math.ceil((index + 1) * 100 / len(file_name))}% (data:{(index + 1)}/{len(file_name)}) -{round((time.time()-start), 2)}s', end='')
    
    # 轉換/合併為numpy陣列
    if len(X_train) != 0:
        X = np.concatenate([X_train, X_tmp])
        Y = np.concatenate([Y_train, Y_tmp])
    else:
        X = np.array(X_tmp)
        Y = np.array(Y_tmp)

    return X, Y

# 讀取單一字母資料(用於字體辨識)
def letter(letter, type):
    X, Y = [], []
    data_file = os.listdir('dataset/{0}'.format(type))
    identifier = 'Capital' if ord(letter) in range(65, 91) else 'Lowercase'
    for file in data_file:
        if identifier in file:
            datas = os.listdir("dataset/{0}/{1}/{2}".format(type, file, letter))
            X, Y = data_append(file, datas, X, Y, type)

    return X, Y
# 讀取整體資料(用於字母辨識)
def all(type):
    all_fonts = []
    X, Y = [], []
    data_file = os.listdir('dataset/{0}'.format(type))

    for file in data_file:
        identifier = 'Capital' if 'Capital' in file else 'Lowercase'
        for i in range(65, 91):
            letter = chr(i+32 if identifier == 'Lowercase' else i)
            datas = os.listdir('dataset/{0}/{1}/{2}'.format(type, file, letter))
            for data in datas:
                all_fonts.append(data)

    X, Y = data_append(data_file, all_fonts, X, Y, type, 'all')

    return X, Y

# 打亂資料
def data_shuffle(X, Y):
    shuffle_ix = np.random.permutation(np.arange(len(Y)))
    X = np.array(X)[shuffle_ix]
    Y = np.array(Y)[shuffle_ix]
    return X, Y

# 呼叫此函式載入目標資料集
def load_data(*args):
    if args == ():  #載入整體資料集
        # 嘗試讀取以儲存的.npy檔
        try:
            X_train = np.load('dataset_array/All_X_train.npy')
            Y_train = np.load('dataset_array/All_Y_train.npy')
            X_test = np.load('dataset_array/All_X_test.npy')
            Y_test = np.load('dataset_array/All_Y_test.npy')
            print('Successful loading of data : All')
        except:
            # 尚未存取則呼叫上面函式載入資料集，並儲存為.npy檔
            try:
                if not os.path.exists('dataset_array/All_X_train.npy'):
                    X_train, Y_train = all('train')
                    print('\nSaving as npy... (All)\n')
                    np.save('dataset_array/All_X_train', X_train)
                    np.save('dataset_array/All_Y_train', Y_train)
                else:
                    X_train = np.load('dataset_array/All_X_train.npy')
                    Y_train = np.load('dataset_array/All_Y_train.npy')
                if not os.path.exists('dataset_array/All_X_test.npy'):
                    X_test, Y_test = all('test')
                    print('\nSaving as npy... (All)\n')
                    np.save('dataset_array/All_X_test', X_test)
                    np.save('dataset_array/All_Y_test', Y_test)
                else:
                    X_test = np.load('dataset_array/All_X_test.npy')
                    Y_test = np.load('dataset_array/All_Y_test.npy')
            except:
                print('Can\'t download data')
    else:
        ASCII = args[0] #載入單一字母資料集
        try:
            X_train = np.load('dataset_array/{0}_X_train.npy'.format(ASCII))
            Y_train = np.load('dataset_array/{0}_Y_train.npy'.format(ASCII))
            X_test = np.load('dataset_array/{0}_X_test.npy'.format(ASCII))
            Y_test = np.load('dataset_array/{0}_Y_tset.npy'.format(ASCII))

            print('Successful loading of data : {0}'.format(chr(ASCII)))
        except:
            try:
                if not os.path.exists('dataset_array/{0}_X_train.npy'.format(ASCII)):
                    X_train, Y_train = letter(chr(ASCII), 'train')
                    print('\nSaving as npy... (letter：{0})'.format(chr(ASCII)))
                    np.save('dataset_array/{0}_X_train'.format(ASCII), X_train)
                    np.save('dataset_array/{0}_Y_train'.format(ASCII), Y_train)
                else:
                    X_train = np.load('dataset_array/{0}_X_train.npy'.format(ASCII))
                    Y_train = np.load('dataset_array/{0}_Y_train.npy'.format(ASCII))
                if not os.path.exists('dataset_array/{0}_X_test.npy'.format(ASCII)):
                    X_test, Y_test = letter(chr(ASCII), 'test')
                    print('\nSaving as npy... (letter：{0})'.format(chr(ASCII)))
                    np.save('dataset_array/{0}_X_test'.format(ASCII), X_test)
                    np.save('dataset_array/{0}_Y_test'.format(ASCII), Y_test)
                else:
                    X_test = np.load('dataset_array/{0}_X_test.npy'.format(ASCII))
                    Y_test = np.load('dataset_array/{0}_Y_test.npy'.format(ASCII))
            except:
                print('Can\'t download data')

    return X_train, Y_train, X_test, Y_test
