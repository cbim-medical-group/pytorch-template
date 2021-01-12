import h5py
import numpy as np

def merge_channels(source1, source2, target, source1_channels, source2_channels, type="train"):
    f_1 = h5py.File(source1, 'r')
    f_2 = h5py.File(source2, 'r')
    keys_1 = list(f_1['train'].keys())
    f_t = h5py.File(target, 'w')
    for key in keys_1:
        source1_data = f_1[f'train/{key}/data'][()]
        source1_label = f_1[f'train/{key}/label'][()]
        source2_data = f_2[f'train/{key}/data'][()]
        target_arr = []
        total_channel = source1_data.shape[0]
        for i in range(total_channel):
            if i in source1_channels:
                target_arr.append(source1_data[i])
            if i in source2_channels:
                target_arr.append(source2_data[i])
        target_data = np.stack(target_arr, axis=0)
        print(f"merge id:{key}")
        f_t.create_dataset(f"train/{key}/data", data=target_data, compression="gzip")
        f_t.create_dataset(f"train/{key}/label", data=source1_label, compression="gzip")

    f_1.close()
    f_2.close()
    f_t.close()

merge_channels("/research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2/General_format_BraTS18_train_2d_3ch_new.h5",
               "/research/cbim/vast/qc58/work/projects/AsynDGANv2/results/brats_AsynDGANv2_3db_exp43_2_doubleemod_t1c/Brats3db_resnet_3ch_doubleemod_t1c/test_100/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch100.h5",
               "/research/cbim/vast/qc58/work/projects/AsynDGANv2/results/brats_AsynDGANv2_3db_exp43_2_doubleemod_t1c/Brats3db_resnet_3ch_doubleemod_t1c/test_100/brats_doublemod_exp46.h5",
               [1,2],[0])


# merge_channels("/research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2/General_format_BraTS18_train_three_center_2d_3ch_new_0.h5",
#                "/research/cbim/vast/qc58/work/projects/AsynDGANv2/results/brats_AsynDGANv2_3db_exp43_2_doubleemod_t1c/Brats3db_resnet_3ch_doubleemod_t1c/test_200/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch200.h5",
#                "/research/cbim/vast/qc58/work/projects/AsynDGANv2/results/brats_AsynDGANv2_3db_exp43_2_doubleemod_t1c/Brats3db_resnet_3ch_doubleemod_t1c/test_200/brats_doublemod_exp46_0.h5",
#                [1,2],[0])







