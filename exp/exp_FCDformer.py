from provide_data.data_loader import Customdataset
from torch.utils.data import Dataset, DataLoader
from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils.tools import EarlyStopping, adjust_learning_rate
import time
import pandas as pd
import numpy as np
import os
from model.models import FIformer, FCDformer
from utils.metrics import metric


class exp_former(Exp_Basic):
    def __init__(self, args):
        super(exp_former, self).__init__(args)
        self.args = args

    def _build_model(self):
        self.device = self._acquire_device()
        self.model = FCDformer(enc_in=self.args.enc_in, dec_in=self.args.dec_in, c_out=self.args.c_out,
                               seq_len=self.args.seq_len, pre_len=self.args.pred_len, fea_num=self.args.fea_num, d_model=self.args.d_model,
                               batch_size=self.args.batch_size, fd_num=self.args.fd_num).to(self.device)
        print(next(self.model.parameters()).is_cuda)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device


    def _get_data(self, flag):
        args = self.args
        data_set = Customdataset(self.args.root_path, self.args.data_path, self.args.target,
                                 flag, self.args.seq_len, self.args.pred_len, self.args.timeenc, self.args.freq)
        data_set.__read_data__()
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            num_workers=0)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, time_x, time_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            time_x = time_x.float().to(self.device)
            time_y = time_y.float().to(self.device)

            f_dim = -1 if self.args.features == 'MS' else 0
            true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            pred = self.model(batch_x, batch_y, time_x, time_y)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self._build_model()

        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.epoch):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # scaler = torch.cuda.amp.GradScaler()

            for i, (batch_x, batch_y, time_x, time_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                time_x = time_x.float().to(self.device)
                time_y = time_y.float().to(self.device)
                iter_count += 1

                model_optim.zero_grad()
                f_dim = -1 if self.args.features == 'MS' else 0
                true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = self.model(batch_x, batch_y, time_x, time_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epoch - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    # scaler.scale(loss).backward()
                    # scaler.step(model_optim)
                    # scaler.update()
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model, best_model_path)
        self.model = torch.load(best_model_path)

        return self.model

    def test(self):
        test_data, test_loader = self._get_data(flag='test')

        # self.device = self._acquire_device()
        #
        # path = '/checkpoints/checkpoint.pth'
        # self.model = torch.load(path)

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, time_x, time_y) in enumerate(test_loader):
            if i == len(test_loader) - 1:
                break
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            time_x = time_x.float().to(self.device)
            time_y = time_y.float().to(self.device)

            f_dim = -1 if self.args.features == 'MS' else 0
            true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            pred = self.model(batch_x, batch_y, time_x, time_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        # preds = preds.tolist()
        # trues = trues.tolist()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, preds.shape[-2], preds.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('mse:{}, mae:{}'.format(mse, mae))
