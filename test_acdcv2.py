import argparse
import importlib

import numpy as np
import os
import plotly.graph_objects as go
import torch
from plotly.offline import plot
from tqdm import tqdm
import h5py
from data_loader.general_data_loader import GeneralDataset
from parse_config import ConfigParser
from skimage.transform import resize


def main(config):
    logger = config.get_logger('test')

    config['data_loader'] = config['test_data_loader']
    # setup data_loader instances
    my_transform = config.init_transform(training=False)
    data_loader = config.init_obj('data_loader', transforms=my_transform, training=True)

    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics

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
    # model = model.cpu()
    model.eval()

    save_dir = os.path.join(config['trainer']['save_dir'], "test_results", config['name'])
    os.makedirs(save_dir, exist_ok=True)
    save_h5 = h5py.File(os.path.join(save_dir, f"test_result-{config['name']}.h5"),"w")
    print(f"Save test results at {save_dir}")

    with torch.no_grad():
        for i, (data, target, misc) in enumerate(tqdm(data_loader)):
            del target
            data = data.squeeze().clone().cpu().numpy()

            if len(data.shape) == 2:
                data = data[:,:,np.newaxis]

            data = resize(data, (320,320), order=1, preserve_range=True)

            tile_data, orig_split_pos, ext_shape = GeneralDataset.split_volume(data, (320, 320, 1), (320, 320, 1))

            tile_outputs = []

            batch_size = 150

            for j in range(tile_data.shape[0]//batch_size+1):
                if j*batch_size == tile_data.shape[0]:
                    # print(f"last tensor is zero... skip")
                    break
                batch_data = torch.tensor(tile_data[j*batch_size:min(tile_data.shape[0], (j+1)*batch_size)])
                batch_data.squeeze_(-1).unsqueeze_(1)

                # print(f"eval:{i} - {j} - {batch_data.shape}")
                output = model(batch_data.cuda().float())
                output.unsqueeze_(-1)
                tile_outputs.append(output.cpu().numpy())

            tile_outputs = np.concatenate(tile_outputs, axis=0)

            # for j in range(tile_data.shape[0]):
            #     input = torch.tensor(tile_data[j]).to(device)
            #     # modify for 2d
            #     input = input.squeeze(-1)
            #     input = input.unsqueeze_(0).unsqueeze_(0)
            #     output = model(input)
            #     output = output.unsqueeze(-1)
            #     tile_outputs.append(output.cpu().numpy())
            #
            # tile_outputs = np.concatenate(tile_outputs, axis=0)

            outputs = []
            for cls_num in range(tile_outputs.shape[1]):
                outputs.append(
                    GeneralDataset.combine_volume(tile_outputs[:, cls_num], data.shape, ext_shape, orig_split_pos,
                                                  (320, 320, 1)))
            output = np.stack(outputs, axis=0)
            new_mask_result = np.argmax(output, 0)

            save_h5.create_dataset(f"{misc['img_path'][0]}", data=data, compression="gzip")
            save_h5.create_dataset(f"{misc['img_path'][0][:-5]}/label", data=new_mask_result, compression="gzip")



            # for patient_id in all_patches_info:
            #     patches = all_patches_info[patient_id]
            #
            #     orig_v_size = patches[0]['orig_size']
            #     orig_v_size = [int(tensor_size) for tensor_size in orig_v_size]
            #     new_mask_result = np.zeros((4, orig_v_size[0], orig_v_size[1], orig_v_size[2]))
            #     for item in patches:
            #         z_index = int(item['z_index'])
            #         slice_result = np.zeros((4, orig_v_size[0], orig_v_size[1]))
            #         result = item['result']
            #         height = min(slice_result.shape[1], result.shape[1])
            #         width = min(slice_result.shape[2], result.shape[2])
            #
            #         slice_result[:, :height, :width] = result[:, :height, :width]
            #
            #         new_mask_result[:, :, :, z_index] = slice_result
            #
            #     new_mask_result = np.argmax(new_mask_result, 0)
            #     save_h5.create_dataset(f"{patient_id}/label", data=new_mask_result)
            #     print(f"Finish rebuild one prediction:{patient_id}...")

        save_h5.close()
        print("Finish rebuild all of predictions")

            #
            # save sample images, or do something with output here
            #
            # pred = torch.argmax(output, 1)
            # pred = (pred == 1)
            # for k in range(data.size(0)):
            #     for t, m, s in zip(data[k], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
            #         t.mul_(s).add_(m)
            #     img_concat = torch.cat((data[k][0], target[k].float(), pred[k].float()), dim=1)
            #     save_img_name = '{:s}/{:s}'.format(save_dir, str(i))
            #     # for i, met in enumerate(metric_fns):
            #         # save_img_name += '-{:s}-{:.4f}'.format(met.__name__, total_metrics[i].item())
            #     save_image(img_concat, save_img_name+"_"+str(k)+'.png')

        if config['trainer']['vis']:
            pred = torch.argmax(output, dim=1).squeeze()
            pred = pred.cpu().numpy()
            pred = pred[:, :, ::-1]
            data = data[:, :, ::-1]
            vis_target = target.squeeze().cpu().numpy()[:, :, ::-1]

            # print(f"before edging:pred sum: {pred.sum()}")
            # pred = edging_mask(pred)
            # data = edging_mask(data)
            # vis_target = edging_mask(vis_target)
            # print(f"after edging, pred sum: {pred.sum()}")

            pred_loc = np.where(pred == 1)
            data_loc = np.where(data == 1)
            target_loc = np.where(vis_target == 1)
            # plot([
            #     go.Mesh3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], alphahull=1, color="lightpink", opacity=0.5),
            #     go.Mesh3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], alphahull=1, color="lightpink", opacity=0.5)
            # ], filename=f'{save_dir}/3dmesh_{str(i)}.html')

            trace1 = go.Scatter3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], mode="markers", marker=dict(size=1),
                                  opacity=0.5, name="pred")
            trace2 = go.Scatter3d(x=data_loc[0], y=data_loc[1], z=data_loc[2], mode="markers", marker=dict(size=1),
                                  opacity=0.5, name="sparse_input")
            trace3 = go.Scatter3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], mode="markers",
                                  marker=dict(size=1),
                                  opacity=0.5, name="target")
            # trace1 = go.Mesh3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], alphahull=1, color='lightpink',
            #                    opacity=0.5, name="pred")
            # trace2 = go.Mesh3d(x=data_loc[0], y=data_loc[1], z=data_loc[2], alphahull=1, color='red',
            #                    opacity=0.5, name="sparse_input")
            # trace3 = go.Mesh3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], alphahull=1, color='blue',
            #                    opacity=0.5, name="target")
            data = [trace1, trace2, trace3]

            layout = go.Layout(
                scene=dict(
                    xaxis=dict(nticks=5, range=[0, 120]),
                    yaxis=dict(nticks=5, range=[0, 120]),
                    zaxis=dict(nticks=5, range=[0, 120])
                )
            )
            fig = go.Figure(data=data, layout=layout)
            plot(fig, filename=f'{save_dir}/Scatter_{str(i)}.html')

            # plot([
            #     go.Scatter3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], mode="markers", marker=dict(size=1),
            #                  opacity=0.5, name="pred"),
            #     go.Scatter3d(x=data_loc[0], y=data_loc[1], z=data_loc[2], mode="markers", marker=dict(size=1),
            #                  opacity=0.5, name="sparse_input"),
            #     go.Scatter3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], mode="markers", marker=dict(size=1),
            #                  opacity=0.5, name="target")
            # ], filename=f'{save_dir}/Scatter3D_{str(i)}.html')
            # computing loss, metrics on test set


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
