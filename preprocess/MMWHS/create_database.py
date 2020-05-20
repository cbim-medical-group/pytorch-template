import h5py
import numpy as np

mmwhs_root = "/Users/qichang/PycharmProjects/medical_dataset/MMWHS/processed"
f1 = h5py.File(f'{mmwhs_root}/MMWHS_train_mpr.h5', 'r')
# f2 = h5py.File('/Users/qichang/PycharmProjects/medical_dataset/ACDC/processed/ACDC_train.h5', 'r')

f3 = h5py.File(f"{mmwhs_root}/MMWHS_training2.h5", "w")
# f4 = h5py.File(f"{mmwhs_root}/ACDC_training.h5", "w")
buf = 15

mean1 = []
std1 = []
mean2 = []
std2 = []
for id in f1['train'].keys():
    print(f"process:{id}")
    data = f1[f"train/{id}/data"][()]
    label = f1[f"train/{id}/label"][()]

    std1.append(np.std(data))
    mean1.append(np.mean(data))

    pos_lvc = np.where(label == 2)
    pos = np.where(label > 0)
    x_min = max(pos_lvc[0].min() - buf, 0)
    x_max = min(pos_lvc[0].max() + buf, label.shape[0])
    y_min = max(pos[1].min() - buf, 0)
    y_max = min(pos[1].max() + buf, label.shape[1])
    z_min = max(pos[2].min() - buf, 0)
    z_max = min(pos[2].max() + buf, label.shape[2])

    data = data[x_min:x_max, y_min: y_max, z_min: z_max]
    data = np.moveaxis(data, 0, -1)
    label = label[x_min:x_max, y_min: y_max, z_min: z_max]
    label = np.moveaxis(label, 0, -1)

    f3.create_dataset(f"train/{id}/data", data=data)
    f3.create_dataset(f"train/{id}/label", data=label)

f3.create_dataset("mean", data=np.average(mean1))
f3.create_dataset("std", data=np.average(std1))
f3.close()

# target_spacing = [1, 1, 10]
#
# for id2 in f2['train'].keys():
#     print(f"2, process: {id2}")
#     for sub_id2 in f2[f"train/{id2}"].keys():
#         print(f"subid: {sub_id2}")
#         if sub_id2.isnumeric():
#             data2 = f2[f"train/{id2}/{sub_id2}/data"][()]
#             label2 = f2[f"train/{id2}/{sub_id2}/label"][()]
#
#             std2.append(np.std(data2))
#             mean2.append(np.mean(data2))
#
#             data2 = np.moveaxis(data2, -1, 0)
#             data2 = data2[:, :, ::-1]
#             label2 = np.moveaxis(label2, -1, 0)
#             label2 = label2[:, :, ::-1]
#
#             x_spacing = f2[f"train/{id2}/{sub_id2}/data"].attrs['x_spacing']
#             y_spacing = f2[f"train/{id2}/{sub_id2}/data"].attrs['y_spacing']
#             z_spacing = f2[f"train/{id2}/{sub_id2}/data"].attrs['z_spacing']
#
#             data2 = ndimage.zoom(data2, (
#                 z_spacing / target_spacing[2], x_spacing / target_spacing[0], y_spacing / target_spacing[1]))
#             label2 = ndimage.zoom(label2, (
#                 z_spacing / target_spacing[2], x_spacing / target_spacing[0], y_spacing / target_spacing[1]), order=0)
#
#             pos_lvc = np.where(label2 == 3)
#             pos = np.where(label2 > 0)
#             x_min = pos_lvc[0].min()
#             x_max = pos_lvc[0].max()
#             y_min = max(pos[1].min() - buf, 0)
#             y_max = min(pos[1].max() + buf, label2.shape[1])
#             z_min = max(pos[2].min() - buf, 0)
#             z_max = min(pos[2].max() + buf, label2.shape[2])
#
#             data2 = data2[:, y_min: y_max, z_min: z_max]
#             data2 = np.moveaxis(data2, 0, -1)
#             label2 = label2[:, y_min: y_max, z_min: z_max]
#             label2 = np.moveaxis(label2, 0, -1)
#             f4.create_dataset(f"train/{id2}_{sub_id2}/data", data=data2)
#             f4.create_dataset(f"train/{id2}_{sub_id2}/label", data=label2)
#
# f4.create_dataset("mean", data=np.average(mean2))
# f4.create_dataset("std", data=np.average(std2))
#
# f4.close()
