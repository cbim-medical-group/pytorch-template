import h5py
import os
import numpy as np

def convert_to_general_data_loader_format(root_path, source_name, target_name, type):
    file = h5py.File(os.path.join(root_path, source_name), 'r')
    target_file = h5py.File(os.path.join(root_path, target_name), 'w')
    for id in list(file['images']):
        print(f"process images id={id}...")
        data = file[f'images/{id}'][()]
        data = np.moveaxis(data,-1,0)
        target_file.create_dataset(f"{type}/{id}/data", data=data)

    for id in list(file['labels']):
        print(f"process label id={id}...")
        target_file.create_dataset(f"{type}/{id}/label", data=file[f'labels/{id}'][()])

    file.close()
    target_file.close()

# convert_to_general_data_loader_format("/share_hd1/db/MNIST/syn_images", "mnist_dcgan_dp_bs256_sigma2_eps4_epoch485_x1.h5", "general_format_mnist_dcgan_dp_bs256_sigma2_eps4_epoch485_x1.h5","train")
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
convert_to_general_data_loader_format("/share_hd1/db/tmp/svhn_dcgan_dp_bs256_sigma2_eps4_256feature/test_160", "cifar10_daddcgan_cDCGAN_epoch160_x1.h5", "general_format_cifar10_daddcgan_cDCGAN_epoch160_x1.h5","train")
# convert_to_general_data_loader_format("/share_hd1/db/tmp/mnist_dcgan_dp_bs256_sigma2_eps4/test_130", "mnist_daddcgan_cDCGAN_epoch130_x1.h5", "general_format_mnist_daddcgan_cDCGAN_epoch130_x1.h5","train")



