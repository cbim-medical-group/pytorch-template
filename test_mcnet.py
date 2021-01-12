import argparse
import importlib

import numpy as np
import os
import plotly.graph_objects as go
import torch
from plotly.offline import plot
from tqdm import tqdm

from data_loader.general_data_loader import GeneralDataset
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    config['data_loader'] = config['test_data_loader']
    # setup data_loader instances
    my_transform = config.init_transform(training=False)
    data_loader = config.init_obj('data_loader', transforms=my_transform, training=False)

    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')
    metric_fns = [getattr(importlib.import_module(f"metric.{met}"), met) for met in config['metric']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    assert config.resume is not None, "In the test script, -r must be assigned to reload well-trained model."
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target, misc) in enumerate(tqdm(data_loader)):
            orig_data, target = data.to(device), target.to(device)

            output = model(orig_data)
            loss = criterion[0](output, target, misc)
            if len(criterion) > 1:
                for idx in range(1, len(criterion)):
                    loss += criterion[idx](output, target, misc)

            batch_size = output.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                misc['input'] = orig_data
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
