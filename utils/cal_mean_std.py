import argparse
import h5py
import numpy as np

def cal_mean_std(file_path: object, type: object = "train") -> object:
    f =  h5py.File(file_path, 'r')
    mean = []
    std = []
    for id in f[type].keys():
        data = f[f'{type}/{id}/data'][()]
        mean.append(data.mean())
        std.append(data.std())
    f.close()
    return np.average(mean), np.std(std)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='mean std tool')
    args.add_argument('-p', '--path', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-t', '--type', default='train', type=str,
                      help='type of dataset (default: train)')

    args = args.parse_args()
    print(args)
    mean, std = cal_mean_std(args.path, args.type)

    print(f"Mean:{mean}, Std:{std}")

#     path:/research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2/General_format_BraTS18_train_random_split_6_1ch.h5

