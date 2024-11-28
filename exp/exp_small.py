
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, get_ema_multi_avg_fn, update_bn
from utils.buffer_large_freq import FastStreamBuffer, SlowStreamBuffer
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_lade_lr, adjust_learning_rate
from utils.metrics import metric, cumavg
from models.lade import Lade, LadeOptimizer

from models_offline import Informer, Autoformer, Transformer,PatchTST, FEDformer, \
      Mvstgn, DLinear, Periodformer, PSLD, FreTS, FourierGNN


from models_offline.stid_arch import stid_arch
from models_offline.dydcrnn_arch import dydcrnn
from models_offline.dydgcrn_arch import dydgcrn
from models_offline.gwnet_arch import gwnet

class Exp(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.device2 = torch.device('cuda:{}'.format(self.args.gpu+1))
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.opt_str = args.opt
        self.n_inner = self._update_strategy()
        self.online_data = None
        self.online_mark = None
        self.criterion = self._select_criterion()
        self.root_path = args.root_path
        self.sample_freq = args.sample_freq
        
        if  'Lade' in self.args.model: 
            self.model = Lade(self.args).float().to(self.device)
            self.opt = LadeOptimizer(self.model, lr=self.args.learning_rate)
        else:
            self.model = self._build_model().to(self.device)
            self.opt = self._select_optimizer()

    def _build_model(self):
        model_dict = {
            'Periodformer': Periodformer,
            'PatchTST': PatchTST,
            'FEDformer':FEDformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'STID':stid_arch,
            'Mvstgn': Mvstgn,
            'DyDcrnn': dydcrnn,
            'DyDgcrn': dydgcrn,
            'Gwnet': gwnet,
            'PSLD': PSLD,
            'FreLinear': FreTS,
            'FourierGNN':FourierGNN,
        }
        
        print('model= ', self.args.model)
        
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag='train'):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), 
                                 lr=self.args.learning_rate, 
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _update_strategy(self):
        return 1 if self.args.sample_freq>=1 else int(1/self.args.sample_freq)

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # if i>1000:break
                iter_count += 1
                x = batch_x.float().to(self.device)
                y = batch_y.float().to(self.device)
                if 'Lade' in self.args.model:
                    f_dim = -1 if self.args.features=='MS' else 0
                    y = y[:,-self.args.pred_len:,f_dim:]
                    _, loss, stati_loss = self.model(x, y, self.criterion)
                    stati_loss.backward()
                else:
                    pred, true =self._get_preds(x,y,i=i)
                    loss = self.criterion(pred, true)
                    
                loss.backward()
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.3f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                self.opt.step()
                self.opt.zero_grad()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valid(vali_loader)
            on_vali_loss = self.test(setting, vali_loader) if self.args.online_valid else vali_loss
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} OnVali Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, on_vali_loss))
            
            early_stopping(on_vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if 'Lade' in self.args.model:
                adjust_lade_lr(self.opt, epoch + 1, self.args)
            else:
                adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def valid(self, vali_loader):
        self.model.eval()
        total_loss = []
        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            x = batch_x.float().to(self.device)
            y = batch_y.float().to(self.device)
            
            pred, true =self._get_preds(x,y,i=t)
            
            loss = self.criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test_loader=None, path=None):
        
        valid = True
        if test_loader is None:
            _, test_loader = self._get_data(flag = 'test')
            valid = False

        if path is not None:
            print('loading model ... ')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + path, 'checkpoint.pth')))

        self.model.eval()
        # ema model
        ema_model = AveragedModel(self.model, device=self.device, multi_avg_fn=get_ema_multi_avg_fn(0.99)) 
        
        # init in another device.
        slow_buffer = SlowStreamBuffer(self.args.pred_len*4+1, self.args.seq_len, self.args.pred_len, self.sample_freq)
        
        # init buffer
        fast_buffer = FastStreamBuffer(self.args.pred_len, self.args.seq_len, self.args.pred_len, self.sample_freq)
        
        preds = []
        trues = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]
        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            # if t>5:break
            x = batch_x.float().to(self.device)
            y = batch_y.float().to(self.device)
            with torch.no_grad():
                pred, true =self._get_preds(x,y,i=t)
            
            # # # 1.1 Slow online updates can be performed on other devices without taking up time from the main process.
            # # # Here, we simply implement it on the same device.
            
            slow_buffer.add_data(batch_x)
            #if slow_buffer.length >=self.args.pred_len: self._slow_online_update_diff(slow_buffer, slow_model, slow_opt)
            if slow_buffer.length > 0: self._slow_online_update_same(slow_buffer, self.model)
                
            # # # 1.2 Update the model online with the latest x. Notice without y!
            fast_buffer.add_data(batch_x, pred.detach().cpu())
            if fast_buffer.length> 0: self._fast_online_update(fast_buffer)
            
            ema_model.update_parameters(self.model)
            
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)
            

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        
        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{:.3f}, mae:{:.3f}, time:{:.2f}'.format(mse, mae, exp_time))
        if valid: return mse
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _get_preds(self, batch_x, batch_y,edge_index=None, edge_attr=None, i=None):
        if 'Decom' in self.args.model:
            x= batch_x.permute(0,2,1)
            outputs = self.model(x, edge_index, edge_attr, batch_y)
            outputs = outputs.permute(0,2,1)
        elif 'TST' in self.args.model:
            outputs = self.model(batch_x.permute(0,2,1), edge_index, edge_attr)
            outputs = outputs.permute(0,2,1)
        elif 'former' in self.args.model \
                or 'Linear' in self.args.model:
            if "Period" in self.args.model or "Fre" in self.args.model:
                outputs = self.model(batch_x)
            else:
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, dec_inp)
        elif 'rnn' in self.args.model:
            if len(batch_x.shape)<3:
                batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.permute(0,2,1)
            outputs = self.model(batch_x, edge_index, edge_attr, batch_y)
        elif "Lade" in self.args.model or "Four" in self.args.model:
            outputs = self.model(batch_x)
        else:
            if len(batch_x.shape)<3:
                batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.permute(0,2,1)
            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            if self.args.model in ('STID','Mvstgn'):
                outputs = self.model(batch_x, edge_index, edge_attr)
            else:
                outputs = self.model(batch_x, edge_index, edge_attr, ycl=None, batch_seen=i)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, batch_y

    
    def _fast_online_update(self,fast_buffer):
        """
            Update the model online with the latest x. Notice without y!
        Args:
            fast_buffer (_type_): lastest samples.
        """
        dataloader = fast_buffer.get_data(self.args.pred_len)
        for _ in range(self.n_inner):
            ### 1. When batch size greater than buffer size.
            # x,x_mark,y,y_mark = next(iter(dataloader))
            
            ### 2. This handles all situations. 
            ### When batch size >= buffer size, The For just execute once.
            for t, (batch_x, batch_y) in enumerate(dataloader):
                # print(x.shape,x_mark.shape,y.shape,y_mark.shape,'---+++')
                x = batch_x.float().to(self.device)
                y = batch_y.float().to(self.device)
                pred, true =self._get_preds(x,y,i=t)
                loss = self.criterion(pred, true)
                loss.backward()
                self.opt.step()       
                self.opt.zero_grad()
        del dataloader

    def _slow_online_update_same(self, slow_buffer, slow_model):
        """
            Slow online updates can be performed on other devices without taking up time from the main process.
            Here, we simply implement it on the same device.
        """
        slow_opt =self.opt
        
        dataloader = slow_buffer.get_data(self.args.pred_len)
        for _ in range(self.n_inner):
            for t, (batch_x, batch_y) in enumerate(dataloader):
                # print(t, batch_x.shape, batch_y.shape, '---+++')
                x = batch_x.float().to(self.device)
                y = batch_y.float().to(self.device)
                pred, true =self._get_preds(x,y,i=t)
                loss = self.criterion(pred, true)
                loss.backward()
                slow_opt.step()
                slow_opt.zero_grad()