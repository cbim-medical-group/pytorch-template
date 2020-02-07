import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import torch.nn.functional as F

import skimage.morphology as morph
from skimage import measure, io

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class NucleiSegTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, test_metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = int(len(data_loader) / 3)

        metric_names = [met.__name__ for met in metric_ftns]
        metric_names.append('loss')
        test_metric_names = [met.__name__ for met in test_metric_ftns]
        test_metric_names.append('loss')
        self.train_metrics = MetricTracker(*metric_names, writer=self.writer)
        self.test_metric_ftns = test_metric_ftns
        self.valid_metrics = MetricTracker(*test_metric_names, writer=self.writer)

        # if config.resume is None:
        #     self.model.initialize()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, weight_map, target, img_name) in enumerate(self.data_loader):
            data, weight_map, target = data.to(self.device), weight_map.to(self.device), target.to(self.device)
            weight_map = weight_map.float().div(20)

            # import matplotlib.pyplot as plt
            # mean = (0.7442, 0.5381, 0.6650)
            # std = (0.1580, 0.1969, 0.1504)
            # for i in range(data.size(0)):
            #     for t, m, s in zip(data[i], mean, std):
            #         t.mul_(s).add_(m)
            #     plt.figure()
            #     plt.imshow(data[i].permute(1,2,0).detach().cpu().numpy())
            #     plt.figure()
            #     plt.imshow(target[i][0].detach().cpu().numpy())
            #     plt.figure()
            #     plt.imshow(weight_map[i][0].detach().cpu().numpy())
            #     plt.show()

            if weight_map.dim() == 4:
                weight_map = weight_map.squeeze(1)
            if target.dim() == 4:
                target = target.squeeze(1)
            if target.max() == 255:
                target /= 255
            # from utils.util import show_figures
            # for i in range(data.size(0)):
            #     show_figures((data[i][0].cpu().numpy(), target[i][0].cpu().numpy()))

            data = data.cuda().detach()

            self.optimizer.zero_grad()
            output = self.model(data)

            log_prob_maps = F.log_softmax(output, dim=1)
            loss_map = self.criterion(log_prob_maps, target, reduction='none')
            loss_map *= weight_map
            loss = loss_map.mean()

            loss.backward()
            self.optimizer.opt.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            self.train_metrics.update('loss', loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, istrain=True), output.size(0))

            if batch_idx % self.log_step == 0:
                message = 'Train Epoch: {} {} Loss: {:.4f}'.format(epoch, self._progress(batch_idx), loss)
                results = self.train_metrics.result()
                for met in self.metric_ftns:
                    met_name = met.__name__
                    message += '\t{:s}: {:.4f}'.format(met_name, results[met_name])
                self.logger.debug(message)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step(epoch)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, weight_map, target, instance_label, img_name) in enumerate(self.valid_data_loader):
                data, weight_map, target = data.to(self.device), weight_map.to(self.device), target.to(self.device)
                weight_map = weight_map.float().div(20)

                if weight_map.dim() == 4:
                    weight_map = weight_map.squeeze(1)
                if target.dim() == 4:
                    target = target.squeeze(1)
                if target.max() == 255:
                    target /= 255

                # output = self.model(data)
                output = self.split_forward(data, 224, 80)
                log_prob_maps = F.log_softmax(output, dim=1)
                loss_map = self.criterion(log_prob_maps, target, reduction='none')
                loss_map *= weight_map
                loss = loss_map.mean()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                pred = torch.argmax(output, dim=1).detach().cpu().numpy()
                pred_inside = pred == 1
                instance_label = instance_label.detach().cpu().numpy()
                for k in range(data.size(0)):
                    pred_inside[k] = morph.remove_small_objects(pred_inside[k], 20)  # remove small object
                    pred[k] = measure.label(pred_inside[k])  # connected component labeling
                    pred[k] = morph.dilation(pred[k], selem=morph.selem.disk(2))
                    instance_label[k] = measure.label(instance_label[k])

                for met in self.test_metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred, instance_label, istrain=False), output.size(0))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def split_forward(self, input, size, overlap, outchannel=3):
        '''
        split the input image for forward process
        '''

        b, c, h0, w0 = input.size()

        # zero pad for border patches
        pad_h = 0
        if h0 - size > 0:
            pad_h = (size - overlap) - (h0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, pad_h, w0)).to(self.device)
            input = torch.cat((input, tmp), dim=2)

        if w0 - size > 0:
            pad_w = (size - overlap) - (w0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, h0 + pad_h, pad_w)).to(self.device)
            input = torch.cat((input, tmp), dim=3)

        _, c, h, w = input.size()

        output = torch.zeros((input.size(0), outchannel, h, w)).to(self.device)
        for i in range(0, h - overlap, size - overlap):
            r_end = i + size if i + size < h else h
            ind1_s = i + overlap // 2 if i > 0 else 0
            ind1_e = i + size - overlap // 2 if i + size < h else h
            for j in range(0, w - overlap, size - overlap):
                c_end = j + size if j + size < w else w

                input_patch = input[:, :, i:r_end, j:c_end]
                input_var = input_patch
                with torch.no_grad():
                    output_patch = self.model(input_var)

                ind2_s = j + overlap // 2 if j > 0 else 0
                ind2_e = j + size - overlap // 2 if j + size < w else w
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                             ind2_s - j:ind2_e - j]

        output = output[:, :, :h0, :w0]

        return output
