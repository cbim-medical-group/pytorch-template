import h5py
import os
import random
from imageio import imread,imsave
from PIL import Image
from scipy.ndimage import rotate
from PIL import Image
import numpy as np

def convert_format():
    img_path = '/Users/qichang/PycharmProjects/notMNIST/open-fonts-dataset/images'
    save_path = '/Users/qichang/PycharmProjects/notMNIST/open-fonts-dataset/new_images'
    filelist = os.listdir(img_path)
    for file in filelist:
        is_ttf = file[-10:-6]
        if is_ttf != ".ttf" and file.endswith(".png"):
            file_path = os.path.join(img_path, file)
            print(f"read:{file_path}")
            font_img = imread(file_path)
            font_img = 255 - font_img
            font_img = np.pad(font_img, 6)
            font_img = np.array(Image.fromarray(font_img).resize((28,28)))
            imsave(os.path.join(save_path, file), font_img)



def list_len_orig():
    train_data = h5py.File('/share_hd1/db/NIPS/MNIST/train_all.h5', 'r')
    test_data = h5py.File('/share_hd1/db/NIPS/MNIST/test.h5', 'r')
    result = {}
    for i in range(10):
        result[i] = 0
    for i in list(test_data['labels']):
        id = test_data[f'labels/{i}'][()]
        result[id] += 1

    print(result)

def build_font_db(da = True):
    font_train_db = h5py.File("/Users/qichang/PycharmProjects/notMNIST/open-fonts-dataset/font_train_db_wo_da.h5", "w")
    font_test_db = h5py.File("/Users/qichang/PycharmProjects/notMNIST/open-fonts-dataset/font_test_db_wo_da.h5", "w")
    img_path = '/Users/qichang/PycharmProjects/notMNIST/open-fonts-dataset/images'
    filelist = os.listdir(img_path)
    train_num = {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
    test_num = {0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982, 5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009}
    file_num = len(filelist)
    train_file_num = round(file_num*0.8)
    for i in range(len(filelist)):
        file = filelist[i]
        if file.endswith(".png"):
            file_path = os.path.join(img_path, file)
            file_id = int(file[-5])
            image = imread(file_path)
            if train_num[file_id] >0:

                font_train_db.create_dataset(f"labels/{file}", data=file_id)
                font_train_db.create_dataset(f"images/{file}", data=image)
                train_num[file_id] -= 1
                print(f"save train:{file}, left:{train_num[file_id]}")

                if da:
                    deg = (random.random()*10-5)
                    r_image = rotate(image, deg, reshape=True)

                    font_train_db.create_dataset(f"labels/da1_{file}", data=file_id)
                    font_train_db.create_dataset(f"images/da1_{file}", data=r_image)
                    train_num[file_id] -= 1

                    deg = (random.random()*10-5)
                    r_image = rotate(image, deg, reshape=True)

                    font_train_db.create_dataset(f"labels/da2_{file}", data=file_id)
                    font_train_db.create_dataset(f"images/da2_{file}", data=r_image)
                    train_num[file_id] -= 1

            elif test_num[file_id] >0:
                font_test_db.create_dataset(f"labels/{file}", data=file_id)
                font_test_db.create_dataset(f"images/{file}", data=image)
                test_num[file_id] -= 1
                print(f"save test:{file}, left:{test_num[file_id]}")

                if da:
                    deg = (random.random()*10-5)
                    r_image = rotate(image, deg, reshape=True)

                    font_test_db.create_dataset(f"labels/da1_{file}", data=file_id)
                    font_test_db.create_dataset(f"images/da1_{file}", data=r_image)
                    train_num[file_id] -= 1

                    deg = (random.random()*10-5)
                    r_image = rotate(image, deg, reshape=True)

                    font_test_db.create_dataset(f"labels/da2_{file}", data=file_id)
                    font_test_db.create_dataset(f"images/da2_{file}", data=r_image)
                    train_num[file_id] -= 1
            else:
                print(f"skip:{file}....")

    font_train_db.close()
    font_test_db.close()



build_font_db()
# convert_format()
