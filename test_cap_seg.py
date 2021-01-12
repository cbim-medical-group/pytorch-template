import argparse
import importlib

import math
import numpy as np
import os
import plotly.graph_objects as go
import torch
import json

from PIL import Image
from plotly.offline import plot
from torchvision.utils import save_image
from tqdm import tqdm
from utils.clean_noise import CleanNoise

from data_loader.general_data_loader import GeneralDataset
from parse_config import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    print(f'==============Configuration==============')
    print(f'{json.dumps(config._config, indent=4)}')
    print(f'==============End Configuration==============')

    logger = config.get_logger('test')

    config['data_loader'] = config['test_data_loader']
    # setup data_loader instances
    my_transform = config.init_transform(training=False)
    data_loader = config.init_obj('data_loader', transforms=my_transform, training=True)

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
    total_metrics_list = [[] for _, fn in enumerate(metric_fns)]

    save_dir = '{:s}/test_results'.format(str(config.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save test results at {save_dir}")

    with torch.no_grad():
        for i, (data, target, misc) in enumerate(tqdm(data_loader)):
            orig_data, target = data.to(device), target.to(device)

            b, c, h, w, d = orig_data.shape
            # crop_x = 192
            # crop_z = 8
            crop_x = 200
            crop_z = 13
            top = (h - crop_x) // 2
            left = (w - crop_x) // 2
            depth = (d - crop_z) // 2

            z_idx_range = list(range(d))
            z_times = math.ceil(d/crop_z)

            output_list = []

            for i in range(z_times):
                z_range = z_idx_range[i*crop_z:min((i+1)*crop_z, d)]
                if len(z_range) < crop_z:
                    z_range = z_idx_range[-crop_z:]
                    clean_data = orig_data[:, :, top: top + crop_x, left: left + crop_x, z_range]
                    output = model(clean_data)
                    output_list.append(output[:,:,:,:,i*crop_z:])
                else:
                    clean_data = orig_data[:, :, top: top + crop_x, left: left + crop_x, z_range]
                    output = model(clean_data)
                    output_list.append(output)

            output = torch.cat(tuple(output_list),-1)
            target = target[:, top: top + crop_x, left: left + crop_x, :]


            # print(clean_data.shape, h, w, d, top, left, depth)

            regression = False

            #convert the output to 2channel binary mask.
            # orig_output = torch.zeros((b, 2, h, w, d)).cuda()
            # if output.shape[1]>2:
            #     #output 4 channels.
            #     orig_output[:, 0] = 0.5
            #     orig_output[:, 1] = 0
            #     output = output[:,0]
            #
            #     new_output = torch.zeros_like(output)
            #     new_output[output<0.5] = 1
            #     orig_output[:, 1, top: top + 200, left: left + 200, depth: depth + 13] = new_output
            #     regression = True
            # else:
            #     orig_output[:, 0] = 20
            #     orig_output[:, 1] = 0
            #     orig_output[:, :2, top: top + 200, left: left + 200, depth: depth + 13] = output[:,:2]
            #     regression = False
            # output = orig_output

            # output = model(orig_data)
            #
            # save sample images, or do something with output here
            #



            # computing loss, metrics on test set
            # if len(target.shape) == 4:
            #     loss = criterion[0](output, target, misc)
            # else:
            #     loss = criterion[0](output, target[:,:2], misc)
            # if len(criterion) > 1:
            #     for idx in range(1, len(criterion)):
            #         loss += criterion[idx](output, target, misc)

            batch_size = output.shape[0]
            # total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                misc['input'] = orig_data
                if len(target.shape) == 4:
                    metric_result = metric(output, target, misc)
                else:
                    metric_result = metric(output, target[:,:2], misc)
                total_metrics[i] += metric_result * batch_size
                total_metrics_list[i].append(metric_result)


            # vis
            idx = [1,5,7,11]
            for k in range(orig_data.size(0)):
                # orig_data[k].mul_(168.6).add_(128)
                o_min = orig_data.min()
                orig_data[k] = (orig_data[k]-o_min)
                orig_data[k] = (orig_data[k] / orig_data[k].max())
                # for t, m, s in zip(output[k], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
                #     t.mul_(s).add_(m)
                # pred = torch.argmax(output, 1)

                #
                pred = output[0]
                pred = torch.argmax(pred, 0)[np.newaxis]
                # clean = CleanNoise()
                # pred_list = []
                # for i in range(pred.shape[2]):
                #     pred_list.append(clean.clean_small_obj(pred[:, :, i].cpu().numpy()))
                # pred = np.stack(pred_list, axis=-1)[np.newaxis]
                # pred = torch.Tensor(pred).cuda()

                target = target

                # convert df to binary
                # bi_output = torch.zeros_like(pred)
                # bi_target = torch.zeros_like(target)
                # bi_output[pred < 0.5] = 1
                # bi_target[target < 0.5] = 1
                # pred = bi_output
                # target = bi_target

                save_img_name = f'{save_dir}/{misc["img_path"][0].split("/")[1]}'

                # target_sample = Image.fromarray(target[k, :,:,idx].cpu().numpy().astype(np.uint8)).convert('RGB')
                # pred_sample = Image.fromarray(pred[k, :,:, idx].cpu().numpy().astype(np.uint8)).convert('RGB')
                # source_sample = Image.fromarray(orig_data[k,0,:,:,idx].cpu().numpy().astype(np.uint8)).convert('RGB')
                # target_sample.save(f"{save_img_name}_{str(k)}_dice:{str(metric_result)}_target.png")
                # pred_sample.save(f"{save_img_name}_{str(k)}_dice:{str(metric_result)}_pred.png")
                # source_sample.save(f"{save_img_name}_{str(k)}_dice:{str(metric_result)}_source.png")

                orig_data_list = []

                for i in idx:
                    if regression:
                        orig_data_list.append(orig_data[k:k + 1, 0, :, :, i])
                        orig_data_list.append(target[k:k + 1,0, :, :, i].float())
                        orig_data_list.append(pred[k:k + 1, :, :, i].float())
                    else:
                        orig_data_list.append(orig_data[k:k+1,0 ,:,:,i])
                        orig_data_list.append(target[k:k+1, :,:,i].float())
                        orig_data_list.append(pred[k:k+1, :,:, i].float())

                img_concat = torch.cat(tuple(orig_data_list), dim=0)

                save_img_name = f"{save_img_name}_{str(k)}_dice:{str(metric_result)}.png"
                # for i, met in enumerate(metric_fns):
                    # save_img_name += '-{:s}-{:.4f}'.format(met.__name__, total_metrics[i].item())
                save_image(img_concat.unsqueeze(1), save_img_name)


    n_samples = len(data_loader.sampler)

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update({
        "detail:"+met.__name__: total_metrics_list[i] for i, met in enumerate(metric_fns)
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
