import h5py
import numpy as np
import os
import random
from scipy import ndimage


def split_train_test_dt_2d(source_path, source_file, target_file_pattern="new_cap_process_#.h5", rate=0.2):
    source = h5py.File(os.path.join(source_path, source_file), 'r')
    keys = list(source['train'].keys())

    train_keys = ['DET0006101', 'DET0030901', 'DET0039401', 'DET0003901', 'DET0003001', 'DET0005201', 'DET0014201', 'DET0004301', 'DET0021501', 'DET0001801', 'DET0044601', 'DET0037101', 'DET0004001', 'DET0001201', 'DET0003301', 'DET0023001', 'DET0004401', 'DET0002701', 'DET0005701', 'DET0042001', 'DET0026801', 'DET0006401', 'DET0008901', 'DET0006201', 'DET0001701', 'DET0002401', 'DET0001501', 'DET0003801', 'DET0001101', 'DET0003501', 'DET0003701', 'DET0024401', 'DET0002801', 'DET0003601', 'DET0009001', 'DET0024501', 'DET0026901', 'DET0015201', 'DET0000801', 'DET0028301', 'DET0042401', 'DET0035501', 'DET0042501', 'DET0043201', 'DET0006001', 'DET0043101', 'DET0005001', 'DET0030101', 'DET0002501', 'DET0010601', 'DET0005901', 'DET0009801', 'DET0016101', 'DET0001301', 'DET0004101', 'DET0021701', 'DET0002001', 'DET0040201', 'DET0043901', 'DET0004201', 'DET0029001', 'DET0039301', 'DET0008801', 'DET0000201', 'DET0004701', 'DET0005401', 'DET0006301', 'DET0001401', 'DET0001601', 'DET0002901', 'DET0000101', 'DET0003201', 'DET0004901', 'DET0028601', 'DET0028801', 'DET0012801', 'DET0040001']
    test_keys = ['DET0042601', 'DET0006501', 'DET0005101', 'DET0009301', 'DET0015401', 'DET0004601', 'DET0003101', 'DET0005601', 'DET0007101', 'DET0043501', 'DET0044801', 'DET0005801', 'DET0014101', 'DET0040101', 'DET0002601', 'DET0039501', 'DET0004801', 'DET0003401', 'DET0015601']
    # case_keys = list(set([key.split("_")[0] for key in keys]))
    # random.shuffle(case_keys)
    # test_size = round(len(case_keys) * rate)
    #
    # train_keys = case_keys[:-test_size]
    # test_keys = case_keys[-test_size:]


    train_h5 = h5py.File(os.path.join(source_path, target_file_pattern.replace('#', 'train')), 'w')
    test_h5 = h5py.File(os.path.join(source_path, target_file_pattern.replace('#', 'test')), 'w')


    for key in list(keys):
        one_case_key = key.split("_")[0]
        if one_case_key in train_keys:
            type = "train"
            h5_file = train_h5
        else:
            type = "test"
            h5_file = test_h5

        sa_label = source[f'train/{key}/sa/label'][()]
        sa_data = source[f'train/{key}/sa/data'][()]

        sa_label[sa_label > 0] = 1

        print(f"save:{key}, sa:{sa_label.shape}")
        for i in range(sa_data.shape[0]):
            sa_label_one = sa_label[i]
            sa_data_one = sa_data[i]

            h5_file.create_dataset(f"{type}/{key}_{i}/data", data=sa_data_one, compression='gzip')
            h5_file.create_dataset(f"{type}/{key}_{i}/label", data=sa_label_one, compression='gzip')

    train_h5.close()
    test_h5.close()

    # for key in list(test_keys):
    #     la_label = source[f'train/{key}/la/label'][()]
    #     sa_label = source[f'train/{key}/sa/label'][()]
    #     la_data = source[f'train/{key}/la/data'][()]
    #     sa_data = source[f'train/{key}/sa/data'][()][np.newaxis, :]
    #     la_label[la_label > 0] = 1
    #     sa_label[sa_label > 0] = 1
    #     sa_arr = []
    #     for i in range(sa_label.shape[0]):
    #         sa_label_one = sa_label[i]
    #         sa_label_dt = ndimage.distance_transform_edt(1 - sa_label_one)
    #         sa_arr.append(sa_label_dt)
    #     sa_label_dt = np.stack(sa_arr)[np.newaxis, :]
    #     print(f"save test:{key}, la:{la_label.shape}, sa:{sa_label_dt.shape}")
    #
    #     label = np.concatenate((sa_label_dt, la_label))
    #     data = np.concatenate((sa_data, la_data))
    #
    #     test_h5.create_dataset(f"test/{key}/data", data=data, compression='gzip')
    #     test_h5.create_dataset(f"test/{key}/label", data=label, compression='gzip')
    #
    # test_h5.close()


split_train_test_dt_2d("/research/cbim/vast/qc58/pub-db/CAP/process/MC-Net", "processed_data.h5","2d_new_cap_process_#.h5")

# la number: [1, 2, 3, 4, 5, 6, 7] : [ 25, 105, 999, 643, 215, 184,  45]
# sa slice numbers: [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24] : [ 75,  73, 243, 398, 496, 227, 390, 135,  95,  44,  20,  20]
# data mean:[128.233673     4.33471628], std:[168.63195145  41.38262084];
# label mean:[8.77938037e+01 8.57572531e-04], std:[5.34328905e+01 2.68363612e-02]