import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker
from pytictoc import TicToc

class TrainerMcnet(BaseTrainer):
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
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        t_iter = TicToc()
        t_epoch = TicToc()
        t_epoch.tic()
        # for batch_idx, (data, target, misc) in enumerate(self.data_loader):
        for batch_idx, enumerate_result in enumerate(self.data_loader):
            t_iter.tic()

            if type(self.data_loader).__name__ == "GeneralDataLoader" or type(self.data_loader).__name__ == "FasterGeneralDataLoader":
                data, target, misc = enumerate_result
            else:
                data, target = enumerate_result
                misc = None

            # DEBUG: save some images for validate
            # for i in range(data.shape[0]):
            #     img_sample = data[i][0]
            #     img_sample = (img_sample * (255/img_sample.max()))
            #     img_sample = img_sample.numpy().astype(np.uint8)
            #     target_sample = target[i]
            #     target_sample = target_sample*255
            #
            #     img_sample = Image.fromarray(img_sample).convert('RGB')
            #     target_sample = Image.fromarray(target_sample.numpy().astype(np.uint8)).convert('RGB')
            #
            #     img_sample.save(f"/ajax/users/qc58/work/projects/pytorch-template/saved/img/{i}-image.jpg")
            #     target_sample.save(f"/ajax/users/qc58/work/projects/pytorch-template/saved/img/{i}-label.jpg")

                # cv2.imwrite(f"/ajax/users/qc58/work/projects/pytorch-template/saved/{i}-image.png",img_sample)
                # cv2.imwrite(f"/ajax/users/qc58/work/projects/pytorch-template/saved/{i}-label.png",target_sample)

            data, target = data.to(self.device), target.to(self.device)
            # print(f"data: {data.shape}, target:{target.shape}")

            self.optimizer.zero_grad()
            output = self.model(data)
            each_loss = []

            output_dt = output[:,0]
            target_dt = target[:,0]

            loss = self.criterion[0](output_dt, target_dt, misc)
            each_loss.append(float(loss))
            if len(self.criterion) > 1:
                for idx in range(1, len(self.criterion)):
                    loss2 = self.criterion[idx](output, target, misc)
                    loss = loss + loss2
                    each_loss.append(float(loss2))
            # loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, misc))

            if batch_idx % self.log_step == 0:
                spam = t_iter.tocvalue(restart=True)
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss:{loss.item():.6f}({each_loss}), elapsed time:{spam:.2f} sec.')
                vis_data = data

                # if data.shape[1] != 3 and data.shape[1] != 1:
                #     vis_data = data[:, 0, :, :, 5]
                #     vis_data = (vis_data - vis_data.min())*(255/vis_data.max()).astype('uint')
                # self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))


            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        spam_epoch = t_epoch.tocvalue(restart=True)
        self.logger.debug(f"Elapsed time of epoch: {spam_epoch:.2f}")

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.logger.debug(f"Learning Rate:{self.lr_scheduler.get_lr()}")
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
            for batch_idx, enumerate_result in enumerate(self.valid_data_loader):
                if type(self.data_loader).__name__ == "GeneralDataLoader" or type(self.data_loader).__name__ == "FasterGeneralDataLoader":
                    data, target, misc = enumerate_result
                else:
                    data, target = enumerate_result
                    misc = None
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                output_dt = output[:, 0]
                target_dt = target[:, 0]

                loss = self.criterion[0](output_dt, target_dt, misc)
                if len(self.criterion) > 1:
                    for idx in range(1, len(self.criterion)):
                        loss += self.criterion[idx](output, target, misc)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, misc))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Validation Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))

                # vis_data=data
                # if data.shape[1] != 3 and data.shape[1] != 1:
                #     vis_data = data[:, 0, :, :, 5]
                #
                # self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
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
