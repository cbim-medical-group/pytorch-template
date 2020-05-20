import os

import h5py
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, binary_erosion
import scipy
from preprocess.MMWHS.mpr import MPR_plane


def calc_bbox(unique):
    return (unique[0].min(), unique[0].max(),
            unique[1].min(), unique[1].max(),
            unique[2].min(), unique[2].max())


def create_prempr_data(root_path):
    f = h5py.File(os.path.join(root_path, 'MMWHS_train.h5'), 'r')
    f2 = h5py.File(os.path.join(root_path, 'MMWHS_train_prempr.h5'), 'w')
    buf = 15
    target_spacing = [1, 1, 1]
    for id in list(f['train']):
        tar_img = f[f"train/{id}/data"][()]
        label = f[f"train/{id}/label"][()]
        tar_img = tar_img[:, ::-1, :]
        label = label[:, ::-1, :]
        target = np.zeros_like(label)
        target[label == 500] = 3
        target[label == 600] = 1
        target[label == 205] = 2
        bbox = calc_bbox(np.where(target > 0))
        #     target = target[max(bbox[0]-buf,0):bbox[1]+buf,
        #                   max(bbox[2]-buf,0):bbox[3]+buf,
        #                   max(bbox[4]-buf,0):bbox[5]+buf]
        #     tar_img = tar_img[max(bbox[0]-buf,0):bbox[1]+buf,
        #                   max(bbox[2]-buf,0):bbox[3]+buf,
        #                   max(bbox[4]-buf,0):bbox[5]+buf]
        #     print("2",[bbox[0]-buf,bbox[1]+buf,
        #                   bbox[2]-buf,bbox[3]+buf,
        #                   bbox[4]-buf,bbox[5]+buf])

        x_spacing, y_spacing, z_spacing = f[f"train/{id}/label"].attrs['x_spacing'], \
                                          f[f"train/{id}/label"].attrs['y_spacing'], f[f"train/{id}/label"].attrs[
                                              'z_spacing']
        print(f"{id}, shape:{f[f'train/{id}/label'].shape}, xspacing,yspacing,zspacing:{x_spacing},{y_spacing},{z_spacing}")
        tar_img = ndimage.zoom(tar_img, (
            x_spacing / target_spacing[0], y_spacing / target_spacing[1], z_spacing / target_spacing[2]))
        target = ndimage.zoom(target, (
            x_spacing / target_spacing[0], y_spacing / target_spacing[1], z_spacing / target_spacing[2]), order=0)

        f2.create_dataset(f"train/{id}/data", data=tar_img)
        f2.create_dataset(f"train/{id}/label", data=target)

    f2.close()

def flood_fill_hull(image):
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull

def creat_mpr(root_path, from_file='MMWHS_train_prempr.h5', save_file="MMWHS_train_mpr.h5"):
    mpr = h5py.File(os.path.join(root_path, from_file), 'r')
    target_data = h5py.File(os.path.join(root_path, save_file), 'w')
    print(f"from:{from_file}| save:{save_file}")
    for i in list(mpr['train'].keys()):
        data = mpr[f"train/{i}/data"][()]
        label = mpr[f"train/{i}/label"][()]

        lvc = np.zeros_like(label)
        lvc[label == 3] = 1
        lvc[label == 2] = 1
        lvc, _ = flood_fill_hull(lvc)

        # lvc_sum = 10000
        # while(lvc_sum>400):
        #     for j in range(lvc.shape[0]):
        #         lvc[j] = binary_erosion(lvc[j])
        #     lvc_sum = lvc.sum()
        # print(f"case {i}, after erosion: {lvc_sum}")

        skeleton_label = skeletonize(lvc)

        if skeleton_label.sum()<1000:
            print(f"The skeleton is shrunk: ({skeleton_label.sum()})...")
            lvc = np.zeros_like(label)
            lvc[label == 3] = 1
            # lvc[label == 2] = 1
            lvc, _ = flood_fill_hull(lvc)
            skeleton_label = skeletonize(lvc)
            print(f"Right now: ({skeleton_label.sum()})...")

        # mesh_verts = np.array(np.where(lvc == 1)).T
        # verts = np.array(np.where(skeleton_label == 255)).T
        verts = np.array(np.where(lvc == 1)).T
        verts_mean = verts.mean(axis=0)
        uu, dd, vv = np.linalg.svd(verts - verts_mean)
        orig_linepts = vv[0] * np.mgrid[-200:200:400j][:, np.newaxis]
        linepts = verts_mean + orig_linepts

        # equal_linepts = get_equidistant_pathline(linepts, 1)

        mpr_labels = []
        mpr_images = []
        for pt in linepts:
            mpr_label, coords = MPR_plane(label, pt, vv[0], [150, 150, 1], order=0)
            if np.sum(mpr_label) ==0:
                continue
            mpr_labels.append(mpr_label)

            mpr_image, coords = MPR_plane(data, pt, vv[0], [150, 150, 1], order=3)
            mpr_images.append(mpr_image)

        mpr_labels_arr = np.stack(mpr_labels, 0)
        mpr_images_arr = np.stack(mpr_images, 0)

        target_data.create_dataset(f'train/{i}/data', data=mpr_images_arr.squeeze())
        target_data.create_dataset(f'train/{i}/label', data=mpr_labels_arr.squeeze())
        print(f"Create training data:{i}, with shape:{mpr_labels_arr.shape}")
    mpr.close()
    target_data.close()



def get_interpoint_dist(pathline):
    """
    calculate distances between each pair of neighboring points
    """
    return np.sqrt(np.sum((pathline[1:] - pathline[:-1]) ** 2, axis=1))


def normalize_vectors(v):
    """
    normalize an array of vectors
    """
    return v / np.sqrt(np.sum(v ** 2, axis=1) + 1e-9).reshape(-1, 1)

def get_equidistant_pathline(pathline, interval):
    """
    create equidistant pathline
    """
    _cur = pathline[0]
    _next = pathline[1]
    newline = [_cur]
    ind = 1

    while True:
        d = get_interpoint_dist(np.array([_cur, _next]))[0]
        v = normalize_vectors((_next - _cur).reshape(1, -1))[0]
        if d >= interval:
            tmp = _cur + v * interval
            newline.append(tmp)
            _cur = tmp
        else:
            ind += 1
            if ind > len(pathline) - 1:
                break
            else:
                _next = pathline[ind]

    return np.array(newline)



# create_prempr_data("/Users/qichang/PycharmProjects/medical_dataset/MMWHS/processed")
creat_mpr("/Users/qichang/PycharmProjects/medical_dataset/MMWHS/processed", from_file='MMWHS_train_prempr.h5',save_file="MMWHS_train_mpr_erosion.h5")
# creat_mpr("/Users/qichang/PycharmProjects/medical_dataset/MMWHS/processed",from_file='MMWHS_train_mpr.h5', save_file="MMWHS_train_mpr_fix.h5")