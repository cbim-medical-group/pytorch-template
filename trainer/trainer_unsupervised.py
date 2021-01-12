import h5py
import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from loss.exp_alignment_mcnet_loss import shift_slice, shift_data_by_offset
from utils import inf_loop, MetricTracker


class TrainerUnsupervised(BaseTrainer):
    """
    Trainer class for unsupervised learning of experiments misalignment correction.

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

        # vis_mcnet = h5py.File(f'saved/test_results/vis_mcnet_2.h5', 'w')

        for batch_idx, (data, target, misc) in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)

            # Test the real data result.
            # for met in self.metric_ftns:
            # met = self.metric_ftns[0]
            # self.train_metrics.update(met.__name__, met(data, target, misc))
            # continue

            # # 2D
            # b, c, w, h, d = data.shape
            # data = data.permute(0, -1, 1, 2, 3)
            # data = data.reshape(b * d, c, w, h)
            # target = target.reshape(b * 10, 2)

            self.optimizer.zero_grad()
            #1
            # output = self.model(data)
            #2
            # zero_y = torch.zeros((data.size(0), 20)).to(self.device)
            # output = self.model(data, zero_y)
            # output = self.model(data, output)
            # output = self.model(data, output)
            #
            # b, w = output.shape
            # reshape_output = output.reshape(b, w // 2, 2)
            # # loss = self.criterion[0](reshape_output, target, misc)
            # #unsupervised:
            # loss = self.criterion[0](reshape_output, data, misc)
            #3
######################################################################
            # shift_range = 10
            # zero_y = torch.zeros((data.size(0),20)).to(self.device)
            # output = self.model(data, zero_y)
            # b, w = output.shape
            #
            # reshape_output = output.reshape(b, w // 2, 2)
            # loss = 0.5*self.criterion[0](reshape_output, target, misc)
            # target = target - reshape_output*0.1
            #
            # #shift1
            # offset = (output * shift_range) / (119/2)
            # data = self.shift_data(offset, data)
            # output = self.model(data, output)
            #
            # reshape_output = output.reshape(b, w // 2, 2)
            # loss += 0.5*self.criterion[0](reshape_output, target, misc)
            # target = target - reshape_output*0.1
            #
            # #shift2
            # offset = (output * shift_range) / (119/2)
            # data = self.shift_data(offset, data)
            # output = self.model(data, output)
            #
            # reshape_output = output.reshape(b, w // 2, 2)
            # loss += self.criterion[0](reshape_output, target, misc)
            # target = target - reshape_output*0.1
            #
            # # output range [-1,1],
            # # output = torch.round(output*10).long()
            #
            # if len(output.shape) == 2:
            #     b, w = output.shape
            #     reshape_output = output.reshape(b, w // 2, 2)
            #     # reshape_output = output.reshape(b, w // 20, 10, 2)
            #
            #     # 2D
            #     # reshape_output = output.reshape(b, 41, 2)
            # else:
            #     reshape_output = output.permute(0, 2, 1)


            # reshape_output.fill_(0)
            # loss = self.criterion[0](target, data, misc)
            # loss = self.criterion[0](reshape_output, data, misc)

            # Recurrent.
            # output = self.model(data, output)
            # reshape_output = output.reshape(b, w // 2, 2)
            # loss += self.criterion[0](reshape_output, target, misc)
            #
            # output = self.model(data, output)
            # reshape_output = output.reshape(b, w // 2, 2)
            # loss += self.criterion[0](reshape_output, target, misc)

            #4
######################################################################

            # shift_range = 10
            # zero_y = torch.zeros((data.size(0),20)).to(self.device)
            # output = self.model(data, zero_y)
            # b, w = output.shape
            #
            # reshape_output = output.reshape(b, w // 2, 2)
            # # loss = 0.5*self.criterion[0](reshape_output, target, misc)
            # target = target - reshape_output
            #
            # #shift1
            # offset = (output * shift_range) / (119/2)
            # data = self.shift_data(offset, data)
            # output = self.model(data, output)
            #
            # reshape_output = output.reshape(b, w // 2, 2)
            # loss = self.criterion[0](reshape_output, target, misc)

#             #5
# ####################################################################
#             shift_range = 10
#             zero_y = torch.zeros((data.size(0),20)).to(self.device)
#             output = self.model(data, zero_y)
#             b, w = output.shape
#
#             reshape_output = output.reshape(b, w // 2, 2)
#             loss = 0.1*self.criterion[0](reshape_output, target, misc)
#             # loss = 0.1*self.criterion[0](reshape_output, data, misc)
#             target = target - reshape_output
#
#             #shift1
#             offset = (output * shift_range) / (119/2)
#             data = self.shift_data(offset, data)
#             output = self.model(data, output)
#
#             reshape_output = output.reshape(b, w // 2, 2)
#             loss += 0.1*self.criterion[0](reshape_output, target, misc)
#             # loss += 0.1*self.criterion[0](reshape_output, data, misc)
#             target = target - reshape_output
#
#             #shift2
#             offset = (output * shift_range) / (119/2)
#             data = self.shift_data(offset, data)
#             output = self.model(data, output)
#
#             reshape_output = output.reshape(b, w // 2, 2)
#             loss += self.criterion[0](reshape_output, target, misc)
#             # loss += self.criterion[0](reshape_output, data, misc)
# #################################################
# 6
####################################################################
            # shift_range = 10
            # zero_y = torch.zeros((data.size(0), 20)).to(self.device)
            #
            # output = self.model(data, zero_y)
            # b, w = output.shape
            # offset = (output * shift_range) / (119 / 2)
            # reshape_offset = offset.reshape(b, w // 2, 2)
            # loss = 0.2 * self.criterion[0](reshape_offset, data, misc)
            # target = target - output.reshape(b, w // 2, 2)
            # # shift1
            # data = self.shift_data(offset, data)
            #
            # output = self.model(data, output)
            # offset = (output * shift_range) / (119 / 2)
            # reshape_offset = offset.reshape(b, w // 2, 2)
            # loss += 0.5 * self.criterion[0](reshape_offset, data, misc)
            # target = target - output.reshape(b, w // 2, 2)
            # # shift2
            # data = self.shift_data(offset, data)
            #
            # output = self.model(data, output)
            # offset = (output * shift_range) / (119 / 2)
            # reshape_offset = offset.reshape(b, w // 2, 2)
            # loss += self.criterion[0](reshape_offset, data, misc)

########
#new exp1
#################################################
            shift_range = 5
            hw_size = 199
            zero_y = torch.zeros((data.size(0), 32)).to(self.device)
            # orig_data = data.clone().cpu().numpy()
            output = self.model(data, zero_y)
            b, w = output.shape
            offset = (output * shift_range) / (hw_size / 2)
            # reshape_offset = offset.reshape(b, w // 2, 2)
            loss = 0.2 * self.criterion[0](offset, data, misc)


            # target = target - output.reshape(b, w // 2, 2)
            # shift1
            data, _, _, _, _ = shift_data_by_offset(offset, data)
            # data = self.shift_data(offset, data)

            output = self.model(data, output)
            offset = (output * shift_range) / (hw_size / 2)
            # reshape_offset = offset.reshape(b, w // 2, 2)
            loss += 0.5 * self.criterion[0](offset, data, misc)
            # target = target - output.reshape(b, w // 2, 2)
            # shift2
            # data = self.shift_data(offset, data)
            data, _, _, _, _ = shift_data_by_offset(offset, data)

            output = self.model(data, output)
            offset = (output * shift_range) / (hw_size / 2)
            # reshape_offset = offset.reshape(b, w // 2, 2)
            loss += self.criterion[0](offset, data, misc)

            # if batch_idx<5:
            #     final_data, _, _, _, _ = shift_data_by_offset(offset, data)
            #     final_data = final_data.clone().cpu().detach().numpy()
            #     vis_mcnet.create_dataset(f"orig_data_{batch_idx}",data=orig_data, compression='gzip')
            #     vis_mcnet.create_dataset(f"fina_data_{batch_idx}",data=final_data, compression='gzip')
            #
            # elif batch_idx==5:
            #     vis_mcnet.close()



#################################################


            # loss = self.criterion[0](reshape_output, target, misc)
            each_loss = []
            # misc['correction_target'] = data
            each_loss.append(float(loss))
            if len(self.criterion) > 1:
                for idx in range(1, len(self.criterion)):
                    loss2 = self.criterion[idx](offset, data, misc)
                    # loss2 = self.criterion[idx](reshape_offset, target, misc)
                    loss = loss + loss2
                    each_loss.append(float(loss2))
            # loss = self.criterion(output, target)
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(data, target, misc))

            #vis
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss:{loss.item():.6f}({each_loss})')

                b, c, w, h, d = data.shape
                vis_data = data.view(b * c, 1, w, h, d).contiguous()[:, :, :, :, 5]
                self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))
                # 2D
                # b, c, w, h = data.shape
                # vis_data = data.reshape(b * c, 1, w, h).contiguous()
                # self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

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
            for batch_idx, (data, target, misc) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Test the real data result.
                # for met in self.metric_ftns:
                # met = self.metric_ftns[0]
                # self.valid_metrics.update(met.__name__, met(data, target, misc))
                # continue

                # # 2D
                # b, c, w, h, d = data.shape
                # data = data.permute(0, -1, 1, 2, 3)
                # data = data.reshape(b * d, c, w, h)
                # target = target.reshape(b * 10, 2)

                #1
                # zero_y = torch.zeros((data.size(0), 20)).to(self.device)
                # output = self.model(data, zero_y)
                # output = self.model(data, output)
                # output = self.model(data, output)
#                 #5
# ###################################################
#                 shift_range = 10
#                 zero_y = torch.zeros((data.size(0), 20)).to(self.device)
#                 output = self.model(data, zero_y)
#                 b, w = output.shape
#
#                 reshape_output = output.reshape(b, w // 2, 2)
#                 loss = 0.1 * self.criterion[0](reshape_output, target, misc)
#                 # loss = 0.1 * self.criterion[0](reshape_output, data, misc)
#                 target = target - reshape_output
#
#                 # shift1
#                 offset = (output * shift_range) / (119 / 2)
#                 data = self.shift_data(offset, data)
#                 output = self.model(data, output)
#
#                 reshape_output = output.reshape(b, w // 2, 2)
#                 loss += 0.1 * self.criterion[0](reshape_output, target, misc)
#                 # loss += 0.1 * self.criterion[0](reshape_output, data, misc)
#                 target = target - reshape_output
#
#                 # shift2
#                 offset = (output * shift_range) / (119 / 2)
#                 data = self.shift_data(offset, data)
#                 output = self.model(data, output)
#
#                 reshape_output = output.reshape(b, w // 2, 2)
#                 loss += self.criterion[0](reshape_output, target, misc)
#                 # loss += self.criterion[0](reshape_output, data, misc)
# ##############################################
                # 6
                ###################################################
                shift_range = 5
                hw_size = 199
                zero_y = torch.zeros((data.size(0), 32)).to(self.device)

                output = self.model(data, zero_y)
                b, w = output.shape
                offset = (output * shift_range) / (hw_size / 2)
                # reshape_offset = offset.reshape(b, w // 2, 2)
                loss = 0.2 * self.criterion[0](offset, data, misc)
                # target = target - output.reshape(b, w // 2, 2)
                #shift1
                data, _, _, _, _ = shift_data_by_offset(offset, data)

                output = self.model(data, output)
                offset = (output * shift_range) / (hw_size / 2)
                # reshape_offset = offset.reshape(b, w // 2, 2)
                loss += 0.5 * self.criterion[0](offset, data, misc)
                # target = target - output.reshape(b, w // 2, 2)
                #shift2
                # data = self.shift_data(offset, data)
                data, _, _, _, _ = shift_data_by_offset(offset, data)

                output = self.model(data, output)
                offset = (output * shift_range) / (hw_size / 2)
                # reshape_offset = offset.reshape(b, w // 2, 2)
                loss += self.criterion[0](offset, data, misc)


                ##############################################

                # output range [-1,1],
                # output = torch.round(output * 10).long()
                # if len(output.shape) == 2:
                #     b, w = output.shape
                #     reshape_output = output.reshape(b, w // 2, 2)
                #     # reshape_output = output.reshape(b, w // 20, 10, 2)
                #
                #     # 2D
                #     # reshape_output = output.reshape(b, 41, 2)
                # else:
                #     reshape_output = output.permute(0, 2, 1)

                # 2D reshape:


                # loss = self.criterion[0](reshape_output, data, misc)
                # loss = self.criterion[0](reshape_output, target, misc)
                # misc['correction_target'] = data
                if len(self.criterion) > 1:
                    for idx in range(1, len(self.criterion)):
                        # loss += self.criterion[idx](reshape_output, data, misc)
                        loss += self.criterion[idx](offset, target, misc)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(data, target, misc))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Validation Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))

                # vis_data = data
                # if data.shape[1] != 3 and data.shape[1] != 1:
                # vis_data = [data[:, 0, :, :, 5].cpu(), data[:, 1, :, :, 5].cpu()]
                b, c, w, h, d = data.shape
                vis_data = data.view(b * c, 1, w, h, d).contiguous()[:, :, :, :, 5]
                self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))

                #2D
                # b, c, w, h = data.shape
                # vis_data = data.reshape(b * c, 1, w, h).contiguous()
                # self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
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
