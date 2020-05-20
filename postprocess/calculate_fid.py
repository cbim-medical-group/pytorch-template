import cv2
import h5py
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil
from postprocess.pytorch_fid.fid_score import calculate_fid_given_paths


def calculate_fid(path, gpu, source_h5_root="images", target_h5_root="images"):
    print("start extracting images....")
    tmp_path = extract_imgs(path[0], path[1], source_h5_root, target_h5_root)
    print("finish extracting images, and start calculate fid....")
    fid_value = calculate_fid_given_paths(tmp_path, 50, gpu, 2048)
    print(f"fid value: {fid_value}")


def extract_imgs(source, target, source_h5_root="images", target_h5_root="images"):
    tmp_path1 = "tmp1"
    tmp_path2 = "tmp2"
    if os.path.isdir(tmp_path1):
        shutil.rmtree(tmp_path1)
    if os.path.isdir(tmp_path2):
        shutil.rmtree(tmp_path2)
    os.mkdir(tmp_path1)
    os.mkdir(tmp_path2)

    source_file = h5py.File(source, 'r')
    target_file = h5py.File(target, 'r')

    img_list1 = list(source_file[f"{source_h5_root}"])
    img_list2 = list(target_file[f"{target_h5_root}"])

    for img_id in img_list1:
        img = source_file[f"{source_h5_root}/{img_id}"][()]
        cv2.imwrite(os.path.join(tmp_path1, img_id + ".png"), img)

    for img_id in img_list2:
        img = target_file[f"{target_h5_root}/{img_id}"][()]
        cv2.imwrite(os.path.join(tmp_path2, img_id + ".png"), img)

    source_file.close()
    target_file.close()

    return [tmp_path1, tmp_path2]


# extract_imgs("/share_hd1/db/Nuclei/lifelong/256/train_all.h5",
#              "/share_hd1/db/Nuclei/lifelong/256/train_D1.h5")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument("source_query",nargs='*', type=str, default="images", help=("source path in h5py"))
parser.add_argument("target_query",nargs='*', type=str, default="images", help=("source path in h5py"))


if __name__ == '__main__':
    args = parser.parse_args()
    calculate_fid(args.path, True, args.source_query, args.target_query)