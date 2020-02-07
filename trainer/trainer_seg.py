import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import torch.nn.functional as F

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class SegTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
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
        self.log_step = int(len(data_loader) / 10)

        metric_names = [met.__name__ for met in metric_ftns]
        metric_names.append('loss')
        self.train_metrics = MetricTracker(*metric_names, writer=self.writer)
        self.valid_metrics = MetricTracker(*metric_names, writer=self.writer)

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
        for batch_idx, (data, target, img_name) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            if target.max() == 255:
                target /= 255
            # from utils.util import show_figures
            # for i in range(data.size(0)):
            #     show_figures((data[i][0].cpu().numpy(), target[i][0].cpu().numpy()))

            data = data.cuda().detach()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(torch.sigmoid(output), target.float())
            loss.backward()
            self.optimizer.opt.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            self.train_metrics.update('loss', loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), output.size(0))

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
            for batch_idx, (data, target, img_name) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                if target.max() == 255:
                    target /= 255
                output = self.model(data)
                loss = self.criterion(torch.sigmoid(output), target.float())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), output.size(0))

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
