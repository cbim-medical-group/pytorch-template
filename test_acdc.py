import argparse

import h5py
import numpy as np
import os
import torch
from tqdm import tqdm

from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    my_transform = config.init_transform()
    data_loader = config.init_obj('data_loader',
                                  transforms=my_transform)

    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )

    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_patches_info = {}
    with torch.no_grad():
        for i, (data, item) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
            item['result'] = output.squeeze().cpu().numpy()

            if str(item['patient_id'][0]) not in all_patches_info:
                all_patches_info[str(item['patient_id'][0])] = []

            all_patches_info[str(item['patient_id'][0])].append(item)
            #
            # save sample images, or do something with output here
            #

    print(f"Finish all patch evaluation....combine and post-processing")
    save_h5 = h5py.File(os.path.join("db", "ACDC", "processed", "Export-1-Cardial_MRI_DB-0-predict-mask.h5"), "w")
    for patient_id in all_patches_info:
        patches = all_patches_info[patient_id]

        orig_v_size = patches[0]['orig_size']
        orig_v_size = [int(tensor_size) for tensor_size in orig_v_size]
        new_mask_result = np.zeros((4, orig_v_size[0], orig_v_size[1], orig_v_size[2]))
        for item in patches:
            z_index = int(item['z_index'])
            slice_result = np.zeros((4, orig_v_size[0], orig_v_size[1]))
            result = item['result']
            height = min(slice_result.shape[1], result.shape[1])
            width = min(slice_result.shape[2], result.shape[2])

            slice_result[:, :height, :width] = result[:, :height, :width]

            new_mask_result[:, :, :, z_index] = slice_result

        new_mask_result = np.argmax(new_mask_result, 0)
        save_h5.create_dataset(f"{patient_id}/label", data=new_mask_result)
        print(f"Finish rebuild one prediction:{patient_id}...")

    save_h5.close()
    print("Finish rebuild all of predictions")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
