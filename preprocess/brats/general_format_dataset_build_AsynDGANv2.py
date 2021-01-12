import os

import numpy as np
import h5py

def general_format(root, source, target, type="1ch", key="train"):
    db = h5py.File(os.path.join(root, source),'r')
    save = h5py.File(os.path.join(root, target), "w")
    for i in db[key].keys():
        t1 = db[f"{key}/{i}/data/t1"][()]
        t2 = db[f"{key}/{i}/data/t2"][()]
        t1ce = db[f"{key}/{i}/data/t1ce"][()]
        flair = db[f"{key}/{i}/data/flair"][()]
        label = db[f"{key}/{i}/label"][()]
        if type == "1ch":
            save.create_dataset(f"{key}/{i}/data", data=t2)
        elif type =="3ch":
            data = np.stack([t1,t2,flair])
            save.create_dataset(f"{key}/{i}/data", data=data)
        elif type =="4ch":
            data = np.stack([t1,t2,flair,t1ce])
            save.create_dataset(f"{key}/{i}/data", data=data)

        save.create_dataset(f"{key}/{i}/label", data=label)

# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_test_2d.h5",
#                "General_format_BraTS18_test_2d_1ch_new.h5",
#                "1ch","test"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_test_2d.h5",
#                "General_format_BraTS18_test_2d_3ch.h5",
#                "3ch","test"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_test_2d.h5",
#                "General_format_BraTS18_test_2d_4ch.h5",
#                "4ch","test"
#                )
#
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_2d.h5",
#                "General_format_BraTS18_train_2d_1ch_new.h5",
#                "1ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_2d.h5",
#                "General_format_BraTS18_train_2d_3ch.h5",
#                "3ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_2d.h5",
#                "General_format_BraTS18_train_2d_4ch.h5",
#                "4ch"
#                )

# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_1_2d.h5",
#                "General_format_BraTS18_train_three_center_1_2d_1ch_new.h5",
#                "1ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_2_2d.h5",
#                "General_format_BraTS18_train_three_center_2_2d_1ch_new.h5",
#                "1ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_3_2d.h5",
#                "General_format_BraTS18_train_three_center_3_2d_1ch_new.h5",
#                "1ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_1_2d.h5",
#                "General_format_BraTS18_train_three_center_1_2d_3ch.h5",
#                "3ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_2_2d.h5",
#                "General_format_BraTS18_train_three_center_2_2d_3ch.h5",
#                "3ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_3_2d.h5",
#                "General_format_BraTS18_train_three_center_3_2d_3ch.h5",
#                "3ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_1_2d.h5",
#                "General_format_BraTS18_train_three_center_1_2d_4ch.h5",
#                "4ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_2_2d.h5",
#                "General_format_BraTS18_train_three_center_2_2d_4ch.h5",
#                "4ch"
#                )
# general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                "BraTS18_train_three_center_3_2d.h5",
#                "General_format_BraTS18_train_three_center_3_2d_4ch.h5",
#                "4ch"
#                )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_0.h5",
               "General_format_BraTS18_train_random_split_0_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_1.h5",
               "General_format_BraTS18_train_random_split_1_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_2.h5",
               "General_format_BraTS18_train_random_split_2_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_3.h5",
               "General_format_BraTS18_train_random_split_3_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_4.h5",
               "General_format_BraTS18_train_random_split_4_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_5.h5",
               "General_format_BraTS18_train_random_split_5_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_6.h5",
               "General_format_BraTS18_train_random_split_6_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_7.h5",
               "General_format_BraTS18_train_random_split_7_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_8.h5",
               "General_format_BraTS18_train_random_split_8_1ch.h5",
               "1ch"
               )

general_format("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
               "BraTS18_train_random_split_9.h5",
               "General_format_BraTS18_train_random_split_9_1ch.h5",
               "1ch"
               )



