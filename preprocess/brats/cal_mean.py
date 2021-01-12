import h5py
import numpy as np

def calc_mean(target):
    f = h5py.File(target, 'r')
    keys = list(f['train'].keys())
    mean_1 = []
    mean_2 = []
    mean_3 = []
    std_1 = []
    std_2 = []
    std_3 = []
    for key in keys:
        data = f[f'train/{key}/data'][()]
        mean_1.append(data[0].mean())
        mean_2.append(data[1].mean())
        mean_3.append(data[2].mean())
        std_1.append(data[0].std())
        std_2.append(data[1].std())
        std_3.append(data[2].std())
        print(f"{key}..")

    mean_1 = np.array(mean_1).mean()
    mean_2 = np.array(mean_2).mean()
    mean_3 = np.array(mean_3).mean()
    std_1 = np.array(std_1).mean()
    std_2 = np.array(std_2).mean()
    std_3 = np.array(std_3).mean()

    print(f"ch1:mean:{mean_1}, std:{std_1}| "
          f"ch2:mean:{mean_2}, std:{std_2}| "
          f"ch3:mean:{mean_3}, std:{std_3}")

calc_mean("/research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2/General_format_BraTS18_train_2d_3ch_new.h5")

# ch1:mean:109.81857307086526, std:198.0402913849317| ch2:mean:107.5048806918867, std:203.84568207898917| ch3:mean:64.17097195910556, std:118.93537507329793

