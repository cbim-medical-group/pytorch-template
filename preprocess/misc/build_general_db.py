import argparse
import os

import h5py
import matplotlib.image as mpimg
import numpy as np


def build_general_db(source_path, target_path, target_name, file_postfix="jpg", db_type="train", comp_type='gzip'):
    item_list = os.listdir(source_path)
    item_list = [item for item in item_list if
                 (os.path.isfile(os.path.join(source_path, item)) and file_postfix in item)]

    target = h5py.File(os.path.join(target_path, target_name), "w")
    for item in item_list:
        data = mpimg.imread(os.path.join(source_path, item))
        if data.shape[0] == 3:
            data = np.moveaxis(data, 0, -1)
        print(f"save item:{item}")
        target.create_dataset(f"{db_type}/{item}/data", data=data, compression=comp_type)
        target.create_dataset(f"{db_type}/{item}/label", data=data, compression=comp_type)

    print(f"finished....")
    target.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate db')
    parser.add_argument("--source", metavar='path', required=True, help="source path")
    parser.add_argument("--target", metavar='path', required=True, help="target apth")
    parser.add_argument("--name", metavar='path', required=True, help="target name")

    parser.add_argument("--file_postfix", metavar='postfix', required=False, help="postfix", default=".jpg")
    parser.add_argument("--db_type", metavar='db_type', required=False, help="db type", default="train")
    parser.add_argument("--comp_type", metavar='comp_type', required=False, help="compression type", default="gzip")

    args = parser.parse_args()
    print(f"execute with args:{args}")
    build_general_db(args.source, args.target, args.name, args.file_postfix, args.db_type, args.comp_type)
