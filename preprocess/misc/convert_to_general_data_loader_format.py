import h5py
import numpy as np
import os


def convert_to_general_data_loader_format(root_path, source_name, target_name, type, convert_label=False,
                                          image_txt="images", label_txt="labels"):
    print(f"Convert the original data /{image_txt}/ and /{label_txt}/ => train|test/ id / data|label")
    file = h5py.File(os.path.join(root_path, source_name), 'r')
    target_file = h5py.File(os.path.join(root_path, target_name), 'w')
    for id in list(file['images']):
        print(f"process images id={id}...")
        data = file[f'{image_txt}/{id}'][()]
        data = np.moveaxis(data, -1, 0)
        target_file.create_dataset(f"{type}/{id}/data", data=data)

    for id in list(file['labels']):
        print(f"process label id={id}...")
        label = file[f'{label_txt}/{id}'][()]
        if convert_label:
            label[label > 0] = 1
        target_file.create_dataset(f"{type}/{id}/label", data=label)

    file.close()
    target_file.close()


# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST", "train_all.h5", "general_format_mnist_train_all.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST", "test.h5", "general_format_mnist_test.h5","val")

# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST", "font_train_db.h5", "general_format_font_train_all.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST", "font_test_db.h5", "general_format_font_test.h5","val")

# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST", "mnist_daddcgan_mix_0_epoch200_x1.h5", "general_format_mnist_daddcgan_mix_0_epoch200_x1.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST", "mnist_daddcgan_mix_0.8_epoch200_x1.h5", "general_format_mnist_daddcgan_mix_0.8_epoch200_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/NIPS/MNIST_font/syn", "mnist_font_mix_daddcgan_epoch200_x1.h5", "general_format_mnist_font_mix_daddcgan_epoch200_x1.h5","train")


# convert_to_general_data_loader_format("/share_hd1/db/MNIST/true_images", "test.h5", "general_format_test.h5","val")

# convert_to_general_data_loader_format("/share_hd1/db/tmp/fashionmnist_dcgan_spectral_norm/test_latest", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/FashionMNIST/true_images", "test.h5", "general_format_test.h5","val")

# convert_to_general_data_loader_format("/share_hd1/db/tmp/mnist_dcgan_dp_bs256_sigma2_eps4/test_latest", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_fashionmnist_dcgan_dp_bs256_sigma2_eps4.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images/", "mnist_dcgan_dp_bs256_sigma2_eps4_epoch485_x1.h5", "general_format_mnist_dcgan_dp_bs256_sigma2_eps4_epoch485_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/tmp/svhn_dcgan_spectral_norm/test_latest", "mnist_daddcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_daddcgan_cDCGAN_epochlatest_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images/ablation_mnist_dcgan_dp_bs256_sigma1_eps4/test_latest", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")
#
# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images/ablation_mnist_dcgan_dp_bs256_sigma1_eps8/test_latest", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")
#
# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images/ablation_mnist_dcgan_dp_bs256_sigma2_eps4/test_latest", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")
#
# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images/ablation_mnist_dcgan_dp_bs256_sigma2_eps8/test_latest", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/tmp/fashionmnist_dcgan_dp_bs256_sigma2_eps4_256ch/test_latest/", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images", "mnist_dcgan_cDCGAN_syn_x2.h5", "general_format_mnist_dcgan_cDCGAN_syn_x2.h5","train")
#
# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images", "mnist_dcgan_cDCGAN_syn_x3.h5", "general_format_mnist_dcgan_cDCGAN_syn_x3.h5","train")
#
# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images", "mnist_dcgan_cDCGAN_syn_x4.h5", "general_format_mnist_dcgan_cDCGAN_syn_x4.h5","train")
#
# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images", "mnist_dcgan_cDCGAN_syn_x5.h5", "general_format_mnist_dcgan_cDCGAN_syn_x5.h5","train")
#

# convert_to_general_data_loader_format("/share_hd1/db/tmp/fashionmnist_dcgan_dp_bs256_sigma2_eps4_256ch/test_latest/", "mnist_dcgan_cDCGAN_epochlatest_x1.h5", "general_format_mnist_dcgan_cDCGAN_epochlatest_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/tmp/fashionmnist_dcgan_dp_bs256_sigma2_eps4_256ch/test_130", "mnist_dcgan_cDCGAN_epoch130_x1.h5", "general_format_mnist_dcgan_cDCGAN_epoch130_x1.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/tmp/svhn_dcgan_dp_bs256_sigma2_eps4_256feature/test_160", "cifar10_daddcgan_cDCGAN_epoch160_x1.h5", "general_format_cifar10_daddcgan_cDCGAN_epoch160_x1.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/tmp/mnist_dcgan_dp_bs256_sigma2_eps4/test_130", "mnist_daddcgan_cDCGAN_epoch130_x1.h5", "general_format_mnist_daddcgan_cDCGAN_epoch130_x1.h5","train")

# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/256", "train_all.h5", "general_format_train_all.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/256", "test.h5", "general_format_test.h5", "val", True)


# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "test_isles_TTP.h5", "general_test_isles_TTP.h5", "val", True)
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "train_isles_TTP.h5", "general_train_isles_TTP.h5", "train", True)

# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "test_isles_MTT.h5", "general_test_isles_MTT.h5", "val", True)
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "train_isles_MTT.h5", "general_train_isles_MTT.h5", "train", True)

# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "test_isles_rCBF.h5", "general_test_isles_rCBF.h5", "val", True)
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "train_isles_rCBF.h5", "general_train_isles_rCBF.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "test_isles_rCBV.h5", "general_test_isles_rCBV.h5", "val", True)
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "train_isles_rCBV.h5", "general_train_isles_rCBV.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "test_isles_Tmax.h5", "general_test_isles_Tmax.h5", "val", True)
# convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images", "train_isles_Tmax.h5", "general_train_isles_Tmax.h5", "train", True)


# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_lifelong", "Brain_lifelong_noDropout_brats_epoch80_task2_brats.h5", "general_Brain_lifelong_noDropout_brats_epoch80_task2_brats.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_lifelong", "Brain_lifelong_noDropout_brats_epoch80_task3_brats.h5", "general_Brain_lifelong_noDropout_brats_epoch80_task3_brats.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_lifelong", "Brain_lifelong_noDropout_brats_epoch80_task3_bratsT1.h5", "general_Brain_lifelong_noDropout_brats_epoch80_task3_bratsT1.h5", "train", True)
#
#
# # convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images/", "test_brats_t1.h5", "general_test_brats_t1.h5", "val", True)
# # convert_to_general_data_loader_format("/share_hd1/db/Brain/real_images/", "test_brats.h5", "general_test_brats.h5", "val", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_finetune", "Brain_finetune_noDropout_brats_epoch80_task2_brats.h5", "general_Brain_finetune_noDropout_brats_epoch80_task2_brats.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_finetune", "Brain_finetune_noDropout_brats_epoch80_task3_brats.h5", "general_Brain_finetune_noDropout_brats_epoch80_task3_brats.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_finetune", "Brain_finetune_noDropout_brats_epoch80_task3_bratsT1.h5", "general_Brain_finetune_noDropout_brats_epoch80_task3_bratsT1.h5", "train", True)
#
#
# # convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_lifelong", "Brain_lifelong_noDropout_brats_epoch80_task2_brats.h5", "general_Brain_lifelong_noDropout_brats_epoch80_task2_brats.h5", "train", True)
# #
# # convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_lifelong", "Brain_lifelong_noDropout_brats_epoch80_task3_brats.h5", "general_Brain_lifelong_noDropout_brats_epoch80_task3_brats.h5", "train", True)
# #
# # convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_lifelong", "Brain_lifelong_noDropout_brats_epoch60_bratsT1.h5", "general_Brain_lifelong_noDropout_brats_epoch60_bratsT1.h5", "train", True)
# #
# #
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_joint", "Brain_joint_noDropout_brats_epoch80_task2_brats.h5", "general_Brain_joint_noDropout_brats_epoch80_task2_brats.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_joint", "Brain_joint_noDropout_brats_epoch80_task3_brats.h5", "general_Brain_joint_noDropout_brats_epoch80_task3_brats.h5", "train", True)
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_joint", "Brain_joint_noDropout_brats_epoch80_task3_bratsT1.h5", "general_Brain_joint_noDropout_brats_epoch80_task3_bratsT1.h5", "train", True)
#
#
# convert_to_general_data_loader_format("/share_hd1/db/Nuclei/lifelong/syn_images_v3/Brain_local", "Brain_local_noDropout_epoch80_bratsT1.h5", "general_Brain_local_noDropout_epoch80_bratsT1.h5", "train", True)


convert_to_general_data_loader_format("/share_hd1/db/NIPS/CIFAR10", "test_cifar10.h5", "general_format_test_cifar10_10label.h5", "val", False, label_txt="fine_labels")
#
# convert_to_general_data_loader_format("/share_hd1/db/NIPS/CIFAR10", "train_cifar10.h5", "general_format_train_cifar10.h5", "train", False)

# convert_to_general_data_loader_format("/share_hd1/db/NIPS/CIFAR10/syn", "cifar10_dcgan_gp_finelabel_cDCGANResnet_v2_epoch150_x1.h5", "general_format_cifar10_dcgan_gp_finelabel_cDCGANResnet_v2_epoch150_x1.h5", "train", False)


