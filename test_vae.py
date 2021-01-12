import argparse

import os

import torch
import importlib

from datetime import datetime
from tqdm import tqdm
from data_loader.general_data_loader import GeneralDataset
from matplotlib import pyplot as plt
from parse_config import ConfigParser
import numpy as np

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )
    my_transform = config.init_transform(training=False)
    data_loader = config.init_obj('data_loader', transforms=my_transform, training=True)


    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')
    metric_fns = [getattr(importlib.import_module(f"metric.{met}"), met) for met in config['metric']]

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

    now = datetime.now()
    save_dir = os.path.join(config['trainer']['save_dir'], "test_results", config['name'], now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))


    with torch.no_grad():
        for i, (data, target, misc) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            vis_target = target.squeeze().cpu().numpy()
            vis_target = (vis_target - np.min(vis_target))/np.ptp(vis_target)
            vis_output = output.squeeze().cpu().numpy()
            vis_output = (vis_output - np.min(vis_output))/np.ptp(vis_output)

            save_img = np.zeros((3, vis_target.shape[1], vis_target.shape[2]*2))
            save_img[:,:,:vis_target.shape[2]] = vis_target
            save_img[:,:,vis_target.shape[2]:] = vis_output

            plt.imsave(os.path.join(save_dir, "test_img_"+str(i)+".png"), np.moveaxis(save_img,0,-1))

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set

            loss = criterion[0](output, target, misc)
            if len(criterion) > 1:
                for idx in range(1, len(criterion)):
                    loss += criterion[idx](output, target, misc)

            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target, misc) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


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
