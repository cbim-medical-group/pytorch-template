import h5py


def cal_mean_std(path, type="train"):
    f = h5py.File(path, "r")
    for key in f[type].keys():
        image = f[f"{type}/image"][()]

